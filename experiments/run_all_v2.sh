#!/usr/bin/env bash
# Every experiment of the paper, in order, from one clean commit. Logs go to results_v2/*.log.
# Benchmark B steps must not share the machine with anything else (timing and memory are reported).
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
R=results_v2
ALL7=overwrite,chunk_lww,incremental,doc_guard,tx_travel,full_history,proposed
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start $(date)" > $R/RUN_ALL.log
# ---- Benchmark A (question set) ----
$PY experiments/run_benchA.py --exp A1_main --seeds 0-9 --holdback 10 --hot_k 2 --ablations 1 > $R/A1_main.log 2>&1
$PY experiments/run_readers.py --exp A1_main > $R/A1_readers.log 2>&1
$PY experiments/run_benchA.py --exp A5_holdback --seeds 0-9 --hot_k 2 --sweep holdback=0,5,10,30,60,120 > $R/A5_holdback.log 2>&1
$PY experiments/run_benchA.py --exp A6_hotk --seeds 0-9 --holdback 10 --sweep hot_k=1,2,4,none > $R/A6_hotk.log 2>&1
echo "A done $(date)" >> $R/RUN_ALL.log
# ---- Benchmark B (real histories) ----
$PY experiments/run_holdback.py --exp B2_holdback --seeds 0,1,2 > $R/B2_holdback.log 2>&1
$PY experiments/run_benchB.py --exp B1_scale --tier_gb 0.25 --ablations 1 --recover 1 > $R/B1_scale_0.25.log 2>&1
$PY experiments/run_holdback.py --exp B2_holdback --gate_tier 0.25 > $R/B2_gate.log 2>&1
$PY experiments/run_benchB.py --exp B1_scale --tier_gb 1 --ablations 1 --recover 1 > $R/B1_scale_1.log 2>&1
$PY experiments/run_concurrency.py --exp B4_concurrency --tier_gb 1 --readers 4 > $R/B4_concurrency.log 2>&1
$PY experiments/run_benchB.py --exp B1_scale --tier_gb 3 --methods $ALL7,full_history_per_revision --recover 1 > $R/B1_scale_3.log 2>&1
$PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 --methods $ALL7,full_history_per_revision --recover 1 > $R/B1_scale_10.log 2>&1
echo "end $(date)" >> $R/RUN_ALL.log
