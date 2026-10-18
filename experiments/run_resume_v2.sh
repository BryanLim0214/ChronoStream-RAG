#!/usr/bin/env bash
# Remaining Benchmark B steps after run_all_v2.sh was stopped by a host memory shortage on 2026-10-17
# (during the 1 GB tier). Everything before that point (A1, readers, A5, A6, B2, B1 0.25 GB, B2 gate)
# finished and is kept; the partial 1 GB tier and B4 were moved to results_v2/_aborted_oom_2026-10-17.
# Same steps and options as run_all_v2.sh. Logs go to results_v2/*.log and RUN_RESUME.log.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
R=results_v2
ALL7=overwrite,chunk_lww,incremental,doc_guard,tx_travel,full_history,proposed
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start $(date)" > $R/RUN_RESUME.log
$PY experiments/run_benchB.py --exp B1_scale --tier_gb 1 --ablations 1 --recover 1 > $R/B1_scale_1.log 2>&1
echo "B1 1 GB done $(date)" >> $R/RUN_RESUME.log
$PY experiments/run_concurrency.py --exp B4_concurrency --tier_gb 1 --readers 4 > $R/B4_concurrency.log 2>&1
echo "B4 done $(date)" >> $R/RUN_RESUME.log
$PY experiments/run_benchB.py --exp B1_scale --tier_gb 3 --methods $ALL7,full_history_per_revision --recover 1 > $R/B1_scale_3.log 2>&1
echo "B1 3 GB done $(date)" >> $R/RUN_RESUME.log
$PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 --methods $ALL7,full_history_per_revision --recover 1 > $R/B1_scale_10.log 2>&1
echo "end $(date)" >> $R/RUN_RESUME.log
