#!/usr/bin/env bash
# Benchmark B scale ladder (B1) + concurrency (B4). Runs sequentially; nothing else should use Qdrant meanwhile.
# Each (method, tier) starts from a restarted Qdrant container. DNFs are recorded, never extrapolated.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
LOG=results_v2/B1_scale.log
ALL7=overwrite,chunk_lww,incremental,doc_guard,tx_travel,full_history,proposed
echo "start $(date)" >> $LOG
$PY experiments/run_benchB.py --exp B1_scale --tier_gb 0.25 --ablations 1 --recover 1 >> $LOG 2>&1
$PY experiments/run_benchB.py --exp B1_scale --tier_gb 1 --ablations 1 --recover 1 >> $LOG 2>&1
$PY experiments/run_concurrency.py --exp B4_concurrency --tier_gb 1 --readers 4 >> results_v2/B4_concurrency.log 2>&1
$PY experiments/run_benchB.py --exp B1_scale --tier_gb 3 --methods $ALL7,full_history_per_revision --recover 1 >> $LOG 2>&1
$PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 --methods $ALL7,full_history_per_revision --recover 1 >> $LOG 2>&1
echo "end $(date)" >> $LOG
