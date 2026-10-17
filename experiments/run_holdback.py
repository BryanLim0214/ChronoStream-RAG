"""B2: holdback trade-off on real Wikipedia revision histories (publication logic only).

python experiments/run_holdback.py --exp B2_holdback --seeds 0,1,2
Outputs results_v2/<exp>/: tradeoff.csv, detector.json, revert_delays.csv, provenance.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from csrag import paths
from csrag.experiment import exposure_and_lag, provenance
from csrag.holdback_sim import publication_replay
from csrag.metrics import build_truth
from csrag.stream import DisorderConfig, simulate_deliveries
from csrag.types import GTRevision, Revision

SWEEP = [0, 10, 30, 60, 120, 300, 600, 1800, 3600, 21600, 86400]


def load_all():
    manifest = json.loads((paths.BENCHB / "manifest.json").read_text(encoding="utf-8"))
    revs, gt, tagrows, delays = [], {}, [], []
    for m in manifest["articles"]:
        t = pq.read_table(paths.BENCHB / "articles" / f"{m['name']}.parquet").to_pylist()
        if len(t) < 2:
            continue
        gt[m["name"]] = []
        for r in t:
            n = len(r["chunk_hashes"])
            revs.append(Revision(m["name"], r["rev_id"], r["t_event"], r["sha1"], ("",) * n, ("",) * n, r["risky"]))
            gt[m["name"]].append(GTRevision(m["name"], r["rev_id"], r["t_event"], r["sha1"], frozenset(), r["reverted"],
                                            r["reverted_tag"]))
            tagrows.append((r["reverted"], r["reverted_tag"], r["risky"]))
            if r["revert_delay_s"] is not None:
                delays.append((r["revert_delay_s"], r["risky"]))
    return revs, gt, tagrows, delays


def gate(tier_gb: float, scale_exp: str, out: Path, dis: DisorderConfig, seed: int = 0):
    """Check the publication-only replay against the full system: on the same tier and delivery
    stream, its head-change log must equal the head log saved by the full B1 runs."""
    from experiments.run_benchB import load_tier
    revs, gt, _ = load_tier(tier_gb)
    deliveries = simulate_deliveries(revs, dis, seed)
    tdir = paths.RESULTS / scale_exp / f"tier_{tier_gb:g}"
    runs = {json.loads(l)["method"]: json.loads(l) for l in open(tdir / "runs.jsonl", encoding="utf-8")}
    checks = []
    for m, rec in runs.items():
        cfg = rec["cfg"]
        f = tdir / f"headlog_{m}.parquet"
        if cfg.get("kind") != "history" or not f.exists() or rec.get("seed") != seed:
            continue
        full = pd.read_parquet(f)
        full_set = {(round(t, 6), d, int(h) if h == h and h is not None else None)
                    for t, d, h in full.itertuples(index=False)}
        sim = publication_replay(deliveries, cfg["holdback_s"], risk_gated=cfg["risk_gated"],
                                 revert_detection=cfg["revert_detection"])
        sim_set = {(round(t, 6), d, h) for t, d, h in sim}
        checks.append({"method": m, "holdback_s": cfg["holdback_s"], "risk_gated": cfg["risk_gated"],
                       "full_events": len(full_set), "sim_events": len(sim_set),
                       "identical": full_set == sim_set, "only_full": len(full_set - sim_set),
                       "only_sim": len(sim_set - full_set)})
        print(checks[-1], flush=True)
    (out / f"gate_tier{tier_gb:g}.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    assert checks and all(c["identical"] for c in checks), "publication replay differs from the full system"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="B2_holdback")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--disorder", default="{}")
    ap.add_argument("--gate_tier", type=float, default=0.0, help="only run the equivalence gate on this B1 tier")
    a = ap.parse_args()
    if a.gate_tier:
        base = dict(p_ooo=0.25, delay_median_s=60, delay_sigma=0.65, dup_rate=0.15, embed_s_per_chunk=0.02,
                    straggler_p=0.28, straggler_scale_s=18, chunk_jitter_p=0.30, chunk_jitter_scale_s=15)
        out = paths.RESULTS / a.exp
        out.mkdir(parents=True, exist_ok=True)
        gate(a.gate_tier, "B1_scale", out, DisorderConfig(**base))
        return
    out = paths.RESULTS / a.exp
    out.mkdir(parents=True, exist_ok=True)
    revs, gt, tagrows, delays = load_all()
    base = dict(p_ooo=0.25, delay_median_s=60, delay_sigma=0.65, dup_rate=0.15, embed_s_per_chunk=0.02,
                straggler_p=0.28, straggler_scale_s=18, chunk_jitter_p=0.30, chunk_jitter_scale_s=15)
    base.update(json.loads(a.disorder))
    dis = DisorderConfig(**base)
    # detector vs MediaWiki's own revert tags (independent ground truth)
    det = np.array([x[0] for x in tagrows]); tag = np.array([x[1] for x in tagrows]); risky = np.array([x[2] for x in tagrows])
    tp = int(np.sum(det & tag))
    detector = {"revisions": len(tagrows), "articles": len(gt), "detected_reverted": int(det.sum()),
                "tagged_mw_reverted": int(tag.sum()), "both": tp,
                "precision_vs_tag": tp / max(det.sum(), 1), "recall_vs_tag": tp / max(tag.sum(), 1),
                "risky_share_all": float(risky.mean()), "risky_share_reverted": float(risky[det].mean()) if det.any() else None,
                "reverted_share": float(det.mean())}
    (out / "detector.json").write_text(json.dumps(detector, indent=1), encoding="utf-8")
    pd.DataFrame(delays, columns=["revert_delay_s", "risky"]).to_csv(out / "revert_delays.csv", index=False)
    print(json.dumps(detector, indent=1), flush=True)
    rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        deliveries = simulate_deliveries(revs, dis, seed)
        truth = build_truth(gt, deliveries)
        end_t = max(max(d.t_avail for d in deliveries), max(r.t_event for r in revs) + max(SWEEP)) + 60
        for H in SWEEP:
            for gated in (False, True):
                if gated and H == 0:
                    continue
                log = publication_replay(deliveries, H, risk_gated=gated)
                el = exposure_and_lag(log, gt, truth, end_t)
                rows.append({"seed": seed, "holdback_s": H, "risk_gated": gated, **el})
                print(f"seed={seed} H={H:6d} gated={gated!s:5s} served={el['reverted_ever_served']}/{el['reverted_revisions']} "
                      f"lag_add_p50={el['lag_added_p50_s']:.0f}s p95={el['lag_added_p95_s']:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out / "tradeoff.csv", index=False)
    (out / "provenance.json").write_text(json.dumps(provenance({"exp": a.exp, "disorder": dis.as_dict(),
                                                                 "sweep": SWEEP}), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
