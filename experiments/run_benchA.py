"""Benchmark A replay: structural probes + retrieval contexts for the readers.

python experiments/run_benchA.py --exp A1_main --seeds 0-9 --methods all
Outputs results_v2/<exp>/: probes.parquet, contexts.parquet, runs.jsonl, provenance.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from csrag import paths
from csrag.embed import Embedder
from csrag.experiment import load_benchA, exposure_and_lag, provenance, file_sha256
from csrag.methods import MethodConfig, make_method, standard_methods
from csrag.metrics import attribute, build_truth, classify, deliverable_target, source_target
from csrag.replay import replay
from csrag.resources import ResourceSampler
from csrag.store import Store
from csrag.stream import DisorderConfig, simulate_deliveries, availability_inversion_rate


def context_state(dt, hs, target):
    """State of the retrieved context (top-k chunks) relative to the QA target revision."""
    if not hs:
        return "empty"
    hset = set(hs)
    if hset & dt.reverted_only:
        return "exposed"
    if target is not None and hset <= target.chunk_set:
        return "clean_target"
    if any(hset <= r.chunk_set for r in dt.revs):
        return "clean_other"
    return "mixed"


def probe_schedule(deliveries, gt, queries, quiesce_s=0.0):
    first_av = {}
    for d in deliveries:
        k = (d.rev.doc_id, d.rev.rev_id)
        first_av[k] = min(first_av.get(k, float("inf")), d.t_avail)
    # final probes come after every delivery AND every holdback timer (quiescent state)
    end_t = max(max(d.t_avail for d in deliveries), max(d.rev.t_event for d in deliveries) + quiesce_s) + 30.0
    probes = []
    for doc, revs in gt.items():
        ts = sorted(first_av[(doc, r.rev_id)] for r in revs)
        times = {t + 0.5 for t in ts}
        times |= {(a + b) / 2 for a, b in zip(ts, ts[1:])}
        for r in revs:
            if r.reverted:
                times |= {r.t_event + 5.0 * k for k in range(7)}
        for t in sorted(times):
            probes.append((t, {"kind": "current", "doc": doc}))
        probes.append((end_t, {"kind": "current_final", "doc": doc}))
    for q in queries:
        probes.append((end_t, {"kind": "asof", "doc": q["doc_id"], "q": q}))
    probes.sort(key=lambda x: x[0])
    return probes, end_t


def run(exp, method_cfgs, seeds, disorder: DisorderConfig, reuse=True, store_text=True):
    out = paths.RESULTS / exp
    out.mkdir(parents=True, exist_ok=True)
    revs, gt, gt_raw, queries = load_benchA()
    emb_q = Embedder(reuse=False)
    cur_q = {doc: g["current_question"] for doc, g in gt_raw.items()}
    qvec_cur = dict(zip(cur_q, emb_q.query(list(cur_q.values()))))
    qvec_asof = dict(zip([q["qid"] for q in queries], emb_q.query([q["question"] for q in queries])))
    gt_by_rev = {(doc, r.rev_id): r for doc, rs in gt.items() for r in rs}
    probe_rows, ctx_rows, run_rows = [], {}, []
    for seed in seeds:
        deliveries = simulate_deliveries(revs, disorder, seed)
        truth = build_truth(gt, deliveries)
        probes, end_t = probe_schedule(deliveries, gt, queries, max(c.holdback_s for c in method_cfgs.values()))
        inv = availability_inversion_rate(deliveries)
        for name, cfg in method_cfgs.items():
            cfg.reuse = reuse
            emb = Embedder(reuse=reuse)
            store = Store(f"{exp}_{name}_{seed}".lower()[:60], store_text=store_text)
            m = make_method(cfg, emb, store, paths.ARCHIVES / exp / name / str(seed))

            def on_probe(method, pr, t):
                doc = pr["doc"]
                dt = truth[doc]
                if pr["kind"] in ("current", "current_final"):
                    V = method.visible(doc, "current")
                    tgt = deliverable_target(dt, t)
                    s = source_target(dt, t)
                    pend = method.docs[doc].pending if (hasattr(method, "docs") and doc in method.docs) else {}
                    held_s = bool(s) and s.rev_id in pend
                    ctx = method.search(doc, qvec_cur[doc], 4, "current")
                    row = {"seed": seed, "method": name, "doc": doc, "t": t, "kind": pr["kind"], "qid": None,
                           "cls_deliverable": classify(dt, V, tgt), "attr": attribute(dt, V, t, held_s),
                           "target_rev": s.rev_id if s else None, "deliverable_rev": tgt.rev_id if tgt else None,
                           "n_visible": len(V)}
                    question = cur_q[doc]
                else:
                    q = pr["q"]
                    V = method.visible(doc, "asof", q["tau"])
                    tgt = gt_by_rev[(doc, q["target_rev"])]
                    ctx = method.search(doc, qvec_asof[q["qid"]], 4, "asof", q["tau"])
                    row = {"seed": seed, "method": name, "doc": doc, "t": t, "kind": q["kind"], "qid": q["qid"],
                           "cls_deliverable": classify(dt, V, tgt), "attr": None,
                           "target_rev": tgt.rev_id, "deliverable_rev": tgt.rev_id, "n_visible": len(V)}
                    question = q["question"]
                hs = [c["chunk_hash"] for c in ctx]
                cid = hashlib.sha256((question + "|" + "|".join(hs)).encode()).hexdigest()[:24]
                if cid not in ctx_rows:
                    ctx_rows[cid] = {"cid": cid, "question": question, "chunk_hashes": hs,
                                     "texts": [c.get("text", "") for c in ctx]}
                row["cid"] = cid
                row["ctx_state"] = context_state(dt, hs, gt_by_rev.get((doc, row["target_rev"])))
                probe_rows.append(row)

            with ResourceSampler() as rs:
                stats = replay(m, deliveries, probes, on_probe)
            el = exposure_and_lag(m.head_log, gt, truth, end_t) if m.head_log else {}
            rec = {"seed": seed, "method": name, "cfg": cfg.as_dict(), "disorder": disorder.as_dict(),
                   "inversion_rate": inv, **stats, **{f"ctr_{k}": v for k, v in m.counters.items()},
                   **{f"emb_{k}": v for k, v in emb.stats().items()},
                   "store_upserted": store.n_upserted, "store_deleted": store.n_deleted,
                   "store_payload_updates": store.n_payload_updates, "store_write_s": store.write_seconds,
                   "final_points": store.count(), **el, **rs.summary(), **m.memory_objects()}
            run_rows.append(rec)
            store.drop()
            print(f"seed={seed} {name:14s} wall={stats['total_wall_s']:.1f}s points={rec['final_points']} "
                  f"exposed_served={el.get('reverted_ever_served')}", flush=True)
    pd.DataFrame(probe_rows).to_parquet(out / "probes.parquet")
    pd.DataFrame(list(ctx_rows.values())).to_parquet(out / "contexts.parquet")
    with open(out / "runs.jsonl", "w", encoding="utf-8") as f:
        for r in run_rows:
            f.write(json.dumps(r, default=str) + "\n")
    prov = provenance({"exp": exp, "seeds": seeds, "methods": {k: v.as_dict() for k, v in method_cfgs.items()},
                       "disorder": disorder.as_dict(),
                       "data_hashes": {f: file_sha256(paths.DATA / "benchA" / f) for f in
                                       ["revisions.jsonl", "gt.json", "queries.json"]}})
    (out / "provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--holdback", type=float, default=60.0)
    ap.add_argument("--hot_k", type=int, default=2)
    ap.add_argument("--reuse", type=int, default=1)
    ap.add_argument("--disorder", default="{}", help="JSON overrides for DisorderConfig")
    ap.add_argument("--ablations", type=int, default=0)
    ap.add_argument("--sweep", default="", help="holdback=0,5,10 | hot_k=1,2,4,none (proposed only)")
    a = ap.parse_args()
    cfgs = standard_methods(a.holdback, a.hot_k)
    if a.ablations:
        cfgs["proposed_no_holdback"] = MethodConfig("proposed_no_holdback", "history", holdback_s=0.0, hot_k=a.hot_k)
        cfgs["proposed_no_revert"] = MethodConfig("proposed_no_revert", "history", holdback_s=a.holdback,
                                                  hot_k=a.hot_k, revert_detection=False)
    if a.sweep:
        key, vals = a.sweep.split("=")
        cfgs = {}
        for v in vals.split(","):
            if key == "holdback":
                cfgs[f"proposed_H{v}"] = MethodConfig(f"proposed_H{v}", "history", holdback_s=float(v), hot_k=a.hot_k)
            elif key == "hot_k":
                k = None if v == "none" else int(v)
                cfgs[f"proposed_K{v}"] = MethodConfig(f"proposed_K{v}", "history", holdback_s=a.holdback, hot_k=k)
    if a.methods != "all":
        cfgs = {k: v for k, v in cfgs.items() if k in a.methods.split(",")}
    run(a.exp, cfgs, parse_seeds(a.seeds), DisorderConfig(**json.loads(a.disorder)), reuse=bool(a.reuse))
