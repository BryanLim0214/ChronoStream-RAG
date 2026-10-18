"""Benchmark B replay: real Wikipedia revision histories at increasing stream sizes.

python experiments/run_benchB.py --exp B1_scale --tier_gb 1 --methods doc_guard,full_history,proposed
Each (method, tier) runs after a Qdrant restart so memory readings are clean.
Outputs results_v2/<exp>/tier_<gb>/: probes.parquet, runs.jsonl, provenance.json
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from csrag import paths
from csrag.embed import Embedder
from csrag.experiment import exposure_and_lag, provenance
from csrag.methods import MethodConfig, ablation_methods, make_method, standard_methods
from csrag.metrics import attribute, build_truth, classify, deliverable_target, source_target
from csrag.replay import replay
from csrag.resources import ResourceSampler
from csrag.revert import detect_reverts
from csrag.store import Store
from csrag.stream import DisorderConfig, availability_inversion_rate, simulate_deliveries
from csrag.types import GTRevision, Revision

NO_WIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def restart_qdrant():
    """Fresh Qdrant process for each policy. While it is stopped, the Linux page cache of Docker's VM is
    dropped, so data cached for earlier policies neither carries over nor keeps holding host memory
    (on 2026-10-17 a cache of ~19 GB in that VM ran the machine out of memory). Failures of the cache
    drop are ignored; it changes no stored data."""
    run = lambda *a: subprocess.run(list(a), capture_output=True, text=True, creationflags=NO_WIN)
    run("docker", "stop", "csrag-qdrant")
    image = run("docker", "inspect", "csrag-qdrant", "--format", "{{.Config.Image}}").stdout.strip()
    if image:
        run("docker", "run", "--rm", "--privileged", "--entrypoint", "sh", image, "-c",
            "sync; echo 3 > /proc/sys/vm/drop_caches")
    run("docker", "start", "csrag-qdrant")
    import urllib.request
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:6333/readyz", timeout=2)
            return
        except Exception:
            time.sleep(0.5)


def load_tier(tier_gb: float):
    """All revisions (any article) whose cumulative raw size in event-time order fits the tier."""
    art_dir = paths.BENCHB / "articles"
    manifest = json.loads((paths.BENCHB / "manifest.json").read_text(encoding="utf-8"))
    index = []
    for m in manifest["articles"]:
        t = pq.read_table(art_dir / f"{m['name']}.parquet", columns=["rev_id", "t_event", "raw_bytes"])
        for rid, te, rb in zip(t["rev_id"].to_pylist(), t["t_event"].to_pylist(), t["raw_bytes"].to_pylist()):
            index.append((te, m["name"], rid, rb))
    index.sort()
    limit = tier_gb * 1e9
    keep, cum = {}, 0
    for te, name, rid, rb in index:
        if cum + rb > limit:
            break
        cum += rb
        keep.setdefault(name, set()).add(rid)
    revs, gt, raw_bytes = [], {}, 0
    for name, rids in keep.items():
        t = pq.read_table(art_dir / f"{name}.parquet").to_pylist()
        chunks = pq.read_table(art_dir / f"{name}.chunks.parquet")
        text = dict(zip(chunks["hash"].to_pylist(), chunks["text"].to_pylist()))
        rows = [r for r in t if r["rev_id"] in rids]
        rows.sort(key=lambda r: (r["t_event"], r["rev_id"]))
        if len(rows) < 2:
            continue
        rev_pos, _ = detect_reverts([r["sha1"] for r in rows])        # ground truth within this replay
        gt[name] = []
        for i, r in enumerate(rows):
            hs = tuple(r["chunk_hashes"])
            revs.append(Revision(name, r["rev_id"], r["t_event"], r["sha1"], tuple(text[h] for h in hs), hs, r["risky"]))
            gt[name].append(GTRevision(name, r["rev_id"], r["t_event"], r["sha1"], frozenset(hs), i in rev_pos,
                                       r["reverted_tag"]))
            raw_bytes += r["raw_bytes"]
    return revs, gt, raw_bytes


def probe_schedule(deliveries, gt, seed, n_random=10, n_exposure=10, n_asof=5, quiesce_s=0.0):
    rng = random.Random(seed)
    first_av = {}
    for d in deliveries:
        k = (d.rev.doc_id, d.rev.rev_id)
        first_av[k] = min(first_av.get(k, float("inf")), d.t_avail)
    end_t = max(max(d.t_avail for d in deliveries), max(d.rev.t_event for d in deliveries) + quiesce_s) + 60.0
    probes = []
    for doc, revs in gt.items():
        avs = sorted(first_av[(doc, r.rev_id)] for r in revs)
        for _ in range(n_random):
            probes.append((rng.uniform(avs[0], avs[-1] + 3600.0), {"kind": "current", "doc": doc}))
        reverted = [r for r in revs if r.reverted]
        for r in rng.sample(reverted, min(n_exposure, len(reverted))):
            probes.append((first_av[(doc, r.rev_id)] + 1.0, {"kind": "current_exposure", "doc": doc}))
        probes.append((end_t, {"kind": "current_final", "doc": doc}))
        cur = [r for r in revs if not r.reverted]
        for _ in range(n_asof):
            tau = rng.uniform(cur[0].t_event, cur[-1].t_event + 1.0)
            probes.append((end_t, {"kind": "asof", "doc": doc, "tau": tau}))
    probes.sort(key=lambda x: x[0])
    return probes, end_t


def run(exp, tier_gb, cfgs, seed, disorder, reuse=True, recover=False, store_text=True, restart=True):
    out = paths.RESULTS / exp / f"tier_{tier_gb:g}"
    out.mkdir(parents=True, exist_ok=True)
    prov0 = provenance()  # recorded at start
    t0 = time.perf_counter()
    revs, gt, raw_bytes = load_tier(tier_gb)
    load_s = time.perf_counter() - t0
    deliveries = simulate_deliveries(revs, disorder, seed)
    truth = build_truth(gt, deliveries)
    probes, end_t = probe_schedule(deliveries, gt, seed, quiesce_s=max(c.holdback_s for c in cfgs.values()))
    inv = availability_inversion_rate(deliveries)
    n_chunks = sum(len(r.chunks) for r in revs)
    print(f"tier={tier_gb}GB articles={len(gt)} revisions={len(revs)} deliveries={len(deliveries)} "
          f"raw_GB={raw_bytes/1e9:.3f} chunk_instances={n_chunks} probes={len(probes)} load_s={load_s:.1f} inv={inv:.4f}", flush=True)
    rows_all = []
    with open(out / "runs.jsonl", "a", encoding="utf-8") as fout:
        for name, cfg in cfgs.items():
            cfg.reuse = reuse
            if restart:
                restart_qdrant()
            emb = Embedder(reuse=reuse)
            store = Store(f"{exp}_{name}_{tier_gb:g}".lower().replace(".", "p")[:60], store_text=store_text)
            m = make_method(cfg, emb, store, paths.ARCHIVES / exp / f"tier_{tier_gb:g}" / name)
            rows = []

            def on_probe(method, pr, t):
                doc = pr["doc"]
                dt = truth[doc]
                if pr["kind"].startswith("current"):
                    V = method.visible(doc, "current")
                    tgt = deliverable_target(dt, t)
                    s = source_target(dt, t)
                    pend = method.docs[doc].pending if (hasattr(method, "docs") and doc in method.docs) else {}
                    rows.append({"method": name, "doc": doc, "t": t, "kind": pr["kind"],
                                 "cls": classify(dt, V, tgt), "attr": attribute(dt, V, t, set(pend)), "n_pending": len(pend),
                                 "n_visible": len(V)})
                else:
                    V = method.visible(doc, "asof", pr["tau"])
                    tgt = source_target(dt, pr["tau"])
                    rows.append({"method": name, "doc": doc, "t": t, "kind": "asof", "tau": pr["tau"],
                                 "cls": classify(dt, V, tgt), "attr": None, "n_visible": len(V)})

            status, err = "ok", None
            try:
                with ResourceSampler(docker_interval=5.0) as rs:
                    stats = replay(m, deliveries, probes, on_probe, progress_every=20000)
                    info = store.info()
                    info["points_exact"] = store.count()          # excludes the barrier sentinel
                el = exposure_and_lag(m.head_log, gt, truth, end_t) if m.head_log else {}
                if m.head_log:
                    pd.DataFrame(m.head_log, columns=["t", "doc", "head"]).to_parquet(out / f"headlog_{name}.parquet")
                rec_stats = m.recover() if (recover and hasattr(m, "recover") and m.archive) else {}
            except Exception as e:  # record a did-not-finish with its reason, never extrapolate
                status, err, stats, info, el, rec_stats = "DNF", repr(e)[:500], {}, {}, {}, {}
                rs = rs if "rs" in dir() else None
            rec = {"method": name, "tier_gb": tier_gb, "status": status, "error": err, "seed": seed,
                   "cfg": cfg.as_dict(), "disorder": disorder.as_dict(), "articles": len(gt), "revisions": len(revs),
                   "deliveries": len(deliveries), "raw_bytes": raw_bytes, "chunk_instances": n_chunks,
                   "inversion_rate": inv, **stats, **{f"ctr_{k}": v for k, v in m.counters.items()},
                   **{f"emb_{k}": v for k, v in emb.stats().items()},
                   "store_upserted": store.n_upserted, "store_deleted": store.n_deleted,
                   "store_payload_updates": store.n_payload_updates, "store_write_s": store.write_seconds,
                   "store_read_retries": store.n_read_retries, **{f"index_{k}": v for k, v in info.items()},
                   "archive_disk_bytes": m.archive.disk_bytes() if m.archive else 0,
                   **el, **(rs.summary() if rs else {}), **m.memory_objects(), **rec_stats}
            if stats:
                rec["throughput_rev_per_s"] = len(deliveries) / max(stats["ingest_wall_s"], 1e-9)
                rec["throughput_raw_MB_per_s"] = raw_bytes / 1e6 / max(stats["ingest_wall_s"], 1e-9)
            fout.write(json.dumps(rec, default=str) + "\n")
            fout.flush()
            if rows:
                pd.DataFrame(rows).to_parquet(out / f"probes_{name}.parquet")
            rows_all += rows
            store.drop()
            print(f"  {name:28s} {status} ingest={stats.get('ingest_wall_s', float('nan')):.1f}s "
                  f"points={info.get('points')} exposed={el.get('reverted_ever_served')}/{el.get('reverted_revisions')} "
                  f"qdrant_res_peak_MB={rec.get('qdrant_memory_resident_bytes_peak', 0)/2**20:.0f}", flush=True)
            del m, emb, store
    # probes are written per method above
    prov = dict(prov0, **{"exp": exp, "tier_gb": tier_gb, "seed": seed, "disorder": disorder.as_dict(),
                       "methods": {k: v.as_dict() for k, v in cfgs.items()},
                       "manifest_totals": json.loads((paths.BENCHB / "manifest.json").read_text())["totals"]})
    (out / "provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--tier_gb", type=float, required=True)
    ap.add_argument("--methods", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdback", type=float, default=600.0)
    ap.add_argument("--hot_k", type=int, default=2)
    ap.add_argument("--reuse", type=int, default=1)
    ap.add_argument("--recover", type=int, default=0)
    ap.add_argument("--ablations", type=int, default=0)
    ap.add_argument("--disorder", default="{}")
    ap.add_argument("--no_restart", type=int, default=0)
    a = ap.parse_args()
    base = dict(p_ooo=0.25, delay_median_s=60, delay_sigma=0.65, dup_rate=0.15, embed_s_per_chunk=0.02,
                straggler_p=0.28, straggler_scale_s=18, chunk_jitter_p=0.30, chunk_jitter_scale_s=15)
    base.update(json.loads(a.disorder))
    cfgs = standard_methods(a.holdback, a.hot_k)
    if a.ablations:
        cfgs.update(ablation_methods(a.holdback, a.hot_k))
        cfgs["proposed_risk_gated"] = MethodConfig("proposed_risk_gated", "history", holdback_s=3600.0,
                                                   risk_gated=True, hot_k=a.hot_k)
    if a.methods != "all":
        cfgs = {k: v for k, v in cfgs.items() if k in a.methods.split(",")}
    run(a.exp, a.tier_gb, cfgs, a.seed, DisorderConfig(**base), reuse=bool(a.reuse), recover=bool(a.recover),
        restart=not a.no_restart)
