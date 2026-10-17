"""Aggregate raw results into paper tables (CSV + JSON). Unit of analysis = document.

python experiments/analyze.py --benchA A1_main [--readers A1_main] [--benchB B1_scale] [--holdback B2_holdback]
Writes results_v2/summary/*.csv and results_v2/summary/numbers.json (every number the paper uses).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from csrag import paths
from csrag.stats import cluster_mean_ci, paired_diff_ci, per_cluster, wilcoxon_holm

SUM = paths.RESULTS / "summary"
SUM.mkdir(parents=True, exist_ok=True)
CLASSES = ["exact", "stale", "torn", "exposed", "missing", "other"]


def ci_cols(df, value_col, cluster_col="doc", prefix=""):
    m, lo, hi = cluster_mean_ci(df[value_col].values, df[cluster_col].values)
    return {f"{prefix}mean": m, f"{prefix}lo": lo, f"{prefix}hi": hi}


def bench_a(exp):
    d = paths.RESULTS / exp
    p = pd.read_parquet(d / "probes.parquet")
    runs = pd.DataFrame([json.loads(l) for l in open(d / "runs.jsonl", encoding="utf-8")])
    cur = p[p.kind.isin(["current", "current_final"])].copy()
    asof = p[p.kind.str.startswith("asof")].copy()
    rows = []
    for m, g in cur.groupby("method"):
        row = {"method": m, "n_probes": len(g)}
        for c in CLASSES:
            g = g.assign(_v=(g.cls_deliverable == c).astype(float))
            row.update(ci_cols(g, "_v", prefix=f"cur_{c}_"))
        a = asof[asof.method == m].assign(_v=lambda x: (x.cls_deliverable == "exact").astype(float))
        row.update(ci_cols(a, "_v", prefix="asof_exact_"))
        av = a[a.kind == "asof_vandal_window"]
        row["asof_vandal_window_exact"] = float(av._v.mean()) if len(av) else float("nan")
        for at in ["correct", "not_arrived", "in_flight", "held", "index_error"]:
            row[f"attr_{at}"] = float((g.attr == at).mean())
        r = runs[runs.method == m]
        for k in ["reverted_ever_served", "reverted_revisions", "lag_added_p50_s", "lag_added_p95_s",
                  "emb_reuse_pct", "final_points", "ctr_retracted", "ctr_suppressed_reverted", "ctr_late_spliced",
                  "ctr_cold_reads", "ingest_wall_s", "inversion_rate"]:
            if k in r:
                row[k] = float(pd.to_numeric(r[k], errors="coerce").mean())
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(SUM / f"{exp}_structural.csv", index=False)
    # paired comparisons on exact share (per-document means)
    pc = {m: per_cluster((g.cls_deliverable == "exact").astype(float).values, g.doc.values)
          for m, g in cur.groupby("method")}
    pa = {m: per_cluster((g.cls_deliverable == "exact").astype(float).values, g.doc.values)
          for m, g in asof.groupby("method")}
    comps = {}
    for other in [m for m in pc if m != "proposed"]:
        d_, lo, hi, nz = paired_diff_ci(pc["proposed"], pc[other])
        da, loa, hia, nza = paired_diff_ci(pa["proposed"], pa[other])
        comps[other] = {"cur_exact_diff": d_, "cur_lo": lo, "cur_hi": hi, "cur_nonzero_docs": nz,
                        "asof_exact_diff": da, "asof_lo": loa, "asof_hi": hia, "asof_nonzero_docs": nza}
    tests = wilcoxon_holm({o: (pc["proposed"], pc[o]) for o in pc if o != "proposed"})
    for o, t in tests.items():
        comps[o].update({f"cur_{k}": v for k, v in t.items()})
    (SUM / f"{exp}_comparisons.json").write_text(json.dumps(comps, indent=1), encoding="utf-8")
    return tab, comps


def readers(exp):
    d = paths.RESULTS / exp
    f = d / "answers_scored.parquet"
    if not f.exists():
        return None
    s = pd.read_parquet(f)
    rows = []
    for (m, rd), g in s.groupby(["method", "reader"]):
        row = {"method": m, "reader": rd, "n": len(g)}
        row.update(ci_cols(g.assign(_v=g.correct.astype(float)), "_v", prefix="acc_"))
        for kind, gk in g.groupby(g.kind.str.startswith("asof")):
            row[f"acc_{'asof' if kind else 'current'}"] = float(gk.correct.mean())
        rows.append(row)
    by_state = s.groupby(["reader", "ctx_state"]).agg(n=("correct", "size"), acc=("correct", "mean"),
                                                       attributed_vandal=("attributed", lambda x: (x.str.startswith("vandal")).mean()),
                                                       attributed_older=("attributed", lambda x: (x.str.startswith("older")).mean())).reset_index()
    pd.DataFrame(rows).to_csv(SUM / f"{exp}_readers.csv", index=False)
    by_state.to_csv(SUM / f"{exp}_readers_by_context.csv", index=False)
    return pd.DataFrame(rows), by_state


def bench_b(exp):
    rows = []
    for tier_dir in sorted(glob.glob(str(paths.RESULTS / exp / "tier_*"))):
        tier = float(Path(tier_dir).name.split("_")[1])
        runs = [json.loads(l) for l in open(Path(tier_dir) / "runs.jsonl", encoding="utf-8")]
        probes = pd.concat([pd.read_parquet(f) for f in glob.glob(str(Path(tier_dir) / "probes_*.parquet"))]) \
            if glob.glob(str(Path(tier_dir) / "probes_*.parquet")) else pd.DataFrame()
        for r in runs:
            row = {"tier_gb": tier, **{k: r.get(k) for k in [
                "method", "status", "articles", "revisions", "deliveries", "raw_bytes", "chunk_instances",
                "ingest_wall_s", "throughput_rev_per_s", "throughput_raw_MB_per_s", "ingest_latency_ms_p50",
                "ingest_latency_ms_p95", "index_points", "index_indexed_vectors", "archive_disk_bytes",
                "py_rss_peak_bytes", "docker_mem_peak_bytes", "qdrant_memory_resident_bytes_peak",
                "qdrant_memory_allocated_bytes_peak", "emb_chunks_encoded", "emb_chunks_requested", "emb_reuse_pct",
                "emb_encode_seconds", "reverted_revisions", "reverted_ever_served", "lag_added_p50_s",
                "lag_added_p95_s", "recovery_serving_ready_s", "recovery_index_green_s", "inversion_rate",
                "ctr_cold_reads", "hot_intervals", "published_revisions"]}}
            if len(probes):
                g = probes[(probes.method == r["method"]) & probes.kind.str.startswith("current")]
                for c in CLASSES:
                    if len(g):
                        row.update(ci_cols(g.assign(_v=(g.cls == c).astype(float)), "_v", prefix=f"cur_{c}_"))
                a = probes[(probes.method == r["method"]) & (probes.kind == "asof")]
                if len(a):
                    row.update(ci_cols(a.assign(_v=(a.cls == "exact").astype(float)), "_v", prefix="asof_exact_"))
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(SUM / f"{exp}_scale.csv", index=False)
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchA")
    ap.add_argument("--readers")
    ap.add_argument("--benchB")
    a = ap.parse_args()
    if a.benchA:
        t, c = bench_a(a.benchA)
        print(t[["method", "cur_exact_mean", "cur_torn_mean", "cur_stale_mean", "cur_exposed_mean", "cur_missing_mean",
                 "asof_exact_mean", "reverted_ever_served", "lag_added_p50_s", "final_points"]].round(3).to_string())
    if a.readers:
        print(readers(a.readers))
    if a.benchB:
        print(bench_b(a.benchB).to_string())
