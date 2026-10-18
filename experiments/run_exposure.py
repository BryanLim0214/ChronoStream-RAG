"""B5: how long readers are exposed to later-undone revisions, and to which kind (review R5).

python experiments/run_exposure.py --exp B5_exposure --seeds 0,1,2
Replays all Benchmark B revisions through the publication rules (csrag/holdback_sim.py; equal to the full
system on the gated tier) and measures, for every later-undone revision, how many seconds it was the
served head. It reports
  * the share of undone revisions ever served and the total exposure time (pre-registered metrics),
  * split by how fast the revert came (within 1 hour or later) and by editor type,
  * with two labels for "undone": our detector, and MediaWiki's own mw-reverted tag (independent truth),
  * for uniform holds, risk-gated holds, and a hold that serves only the newest arrived revision of a
    document (supersede; no revert detector; simulator only).
Outputs results_v2/<exp>/exposure.csv, freshness.csv, provenance.json.
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
from csrag.experiment import exposure_and_lag, head_intervals, provenance
from csrag.holdback_sim import publication_replay
from csrag.metrics import build_truth
from csrag.stream import DisorderConfig, simulate_deliveries
from experiments.run_holdback import load_all

CONFIGS = [  # (name, H seconds, risk_gated, revert_detection, supersede)
    ("none", 0, False, True, False),
    ("uniform_10min", 600, False, True, False),
    ("uniform_1h", 3600, False, True, False),
    ("uniform_6h", 21600, False, True, False),
    ("uniform_1day", 86400, False, True, False),
    ("gated_1h", 3600, True, True, False),
    ("gated_1day", 86400, True, True, False),
    ("supersede_10min", 600, False, False, True),
    ("supersede_1h", 3600, False, False, True),
    ("no_detector_10min", 600, False, False, False),
]
FAST_S = 3600.0


def revision_facts():
    """(doc, rev_id) -> revert delay, editor without account, detector label, MediaWiki tag."""
    manifest = json.loads((paths.BENCHB / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for m in manifest["articles"]:
        t = pq.read_table(paths.BENCHB / "articles" / f"{m['name']}.parquet",
                          columns=["rev_id", "revert_delay_s", "risky", "reverted", "reverted_tag"]).to_pylist()
        for r in t:
            out[(m["name"], r["rev_id"])] = r
    return out


def exposure_rows(log, facts, end_t):
    """Seconds each revision was the served head, for every revision labelled undone by either label."""
    iv = head_intervals(log, end_t)
    secs = {}
    for doc, ivs in iv.items():
        for s, e, r in ivs:
            if r is not None:
                secs[(doc, r)] = secs.get((doc, r), 0.0) + (e - s)
    rows = []
    for key, f in facts.items():
        if not (f["reverted"] or f["reverted_tag"]):
            continue
        rows.append((key, f["reverted"], f["reverted_tag"], f["risky"], f["revert_delay_s"], secs.get(key, 0.0)))
    return pd.DataFrame(rows, columns=["key", "det", "tag", "risky", "delay_s", "exposure_s"])


def summarize(df):
    out = []
    for label in ["det", "tag"]:
        d = df[df[label]]
        strata = {"all": d, "registered": d[~d.risky], "no_account": d[d.risky]}
        if label == "det":   # revert delay is known for detector-labelled revisions
            strata["revert_within_1h"] = d[d.delay_s <= FAST_S]
            strata["revert_after_1h"] = d[d.delay_s > FAST_S]
        for name, g in strata.items():
            out.append({"label": label, "stratum": name, "revisions": len(g), "served": int((g.exposure_s > 0).sum()),
                        "served_pct": 100.0 * (g.exposure_s > 0).mean() if len(g) else np.nan,
                        "exposure_h": g.exposure_s.sum() / 3600.0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="B5_exposure")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    out = paths.RESULTS / a.exp
    out.mkdir(parents=True, exist_ok=True)
    revs, gt, _, _ = load_all()
    facts = revision_facts()
    dis = DisorderConfig(p_ooo=0.25, delay_median_s=60, delay_sigma=0.65, dup_rate=0.15, embed_s_per_chunk=0.02,
                         straggler_p=0.28, straggler_scale_s=18, chunk_jitter_p=0.30, chunk_jitter_scale_s=15)
    rows, fresh = [], []
    for seed in [int(s) for s in a.seeds.split(",")]:
        deliveries = simulate_deliveries(revs, dis, seed)
        truth = build_truth(gt, deliveries)
        end_t = max(max(d.t_avail for d in deliveries), max(r.t_event for r in revs) + 86400) + 60
        for name, H, gated, det, sup in CONFIGS:
            log = publication_replay(deliveries, H, risk_gated=gated, revert_detection=det, supersede=sup)
            df = exposure_rows(log, facts, end_t)
            for r in summarize(df):
                rows.append({"seed": seed, "config": name, **r})
            el = exposure_and_lag(log, gt, truth, end_t)
            fresh.append({"seed": seed, "config": name, **{k: el[k] for k in
                          ["lag_added_mean_s", "lag_added_p50_s", "lag_added_p95_s", "heads_never_served", "heads_considered"]}})
            a_ = [r for r in rows if r["seed"] == seed and r["config"] == name and r["label"] == "det" and r["stratum"] == "all"][0]
            print(f"seed={seed} {name:18s} served={a_['served_pct']:.1f}% exposure_h={a_['exposure_h']:.0f}", flush=True)
    pd.DataFrame(rows).to_csv(out / "exposure.csv", index=False)
    pd.DataFrame(fresh).to_csv(out / "freshness.csv", index=False)
    (out / "provenance.json").write_text(json.dumps(provenance({"exp": a.exp, "disorder": dis.as_dict(),
                                                                 "configs": CONFIGS, "fast_s": FAST_S}), indent=1),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
