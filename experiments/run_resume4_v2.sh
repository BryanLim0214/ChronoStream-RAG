#!/usr/bin/env bash
# Fourth schedule (2026-10-17, about 22:40, at the user's request to go faster once the game had closed).
# The B4 worker from run_resume3_v2.sh is still running on csrag-qdrant-3, and the last 1 GB ablation on
# csrag-qdrant-4. The 10 GB tier is split over two workers: ChronoStream-RAG and Full history start now,
# Document guard starts once the 1 GB runs have freed memory. None of these runs reports throughput.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
R=results_v2
ALL7=overwrite,chunk_lww,incremental,doc_guard,tx_travel,full_history,proposed
log() { echo "$* $(date)" >> $R/RUN_RESUME4.log; }
free_gb() { awk '/MemFree/ {print int($2/1048576)}' /proc/meminfo; }
need() { while [ "$(free_gb)" -lt "$1" ]; do sleep 30; done; }
w() { env CSRAG_WORKER=$1 CSRAG_QDRANT_HTTP=$2 CSRAG_QDRANT_GRPC=$3 CSRAG_QDRANT_CONTAINER=$4 "${@:5}"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start $(date)" > $R/RUN_RESUME4.log

need 14
w w1 6333 6334 csrag-qdrant $PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 \
    --methods proposed,full_history --recover 1 > $R/B1_scale_10_w1.log 2>&1 &
P1=$!; log "10 GB w1 (proposed, full_history) started"
sleep 300; need 7
w w2 6335 6336 csrag-qdrant-2 $PY experiments/run_benchB.py --exp B1_stress --tier_gb 1 --methods $ALL7 \
    --disorder '{"delay_median_s": 600}' > $R/B1_stress_1.log 2>&1 &
P2=$!; log "stress started"

# seed 1 on csrag-qdrant-4 once the last 1 GB ablation has written its line
while [ "$(lines $R/B1_scale/tier_1/runs_w4.jsonl)" -lt 4 ]; do sleep 30; done
sleep 60; need 7
w w4 6339 6340 csrag-qdrant-4 $PY experiments/run_benchB.py --exp B1_seed1 --tier_gb 1 --methods $ALL7 --seed 1 \
    > $R/B1_seed1_1.log 2>&1 &
P4=$!; log "seed 1 started"

wait $P2; log "stress finished"
need 7
w w2 6335 6336 csrag-qdrant-2 $PY experiments/run_benchB.py --exp B1_seed2 --tier_gb 1 --methods $ALL7 --seed 2 \
    > $R/B1_seed2_1.log 2>&1 &
P2=$!; log "seed 2 started"

wait $P4; log "seed 1 finished"
need 14
w w5 6339 6340 csrag-qdrant-4 $PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 \
    --methods doc_guard --recover 1 > $R/B1_scale_10_w5.log 2>&1 &
P5=$!; log "10 GB w5 (doc_guard) started"

wait $P2; log "seed 2 finished"
wait $P1; log "10 GB w1 finished"
wait $P5; log "10 GB w5 finished"
log "end"
