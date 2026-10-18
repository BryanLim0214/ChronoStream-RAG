#!/usr/bin/env bash
# Third schedule (2026-10-17, about 21:50). Replaces phases 3-5 of run_resume2_v2.sh, whose shell was paused
# while its 3 GB worker (Document guard, Full history, ChronoStream-RAG on csrag-qdrant-2) kept running.
# Changes:
#   - B4 and the 1 GB ablations start now, beside the 3 GB worker, on their own Qdrant containers.
#   - Whole-revision Full history is dropped from the 1 GB tier (it is reported at 0.25 GB only); it needed
#     more memory than the host could spare beside the other workers.
#   - The 10 GB tier runs as one worker (two would need about twice the Python memory), most important
#     policies first, and starts as soon as the 3 GB worker ends.
#   - The stress setting and the two extra seeds run one after another on one container.
# None of these runs reports throughput or recovery time; index states, stored vectors and each container's
# own memory metric do not depend on other workers. Each worker starts only when enough memory is free.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
R=results_v2
ALL7=overwrite,chunk_lww,incremental,doc_guard,tx_travel,full_history,proposed
ABL=proposed_no_holdback,proposed_no_revert,proposed_inplace_commit,proposed_risk_gated
log() { echo "$* $(date)" >> $R/RUN_RESUME3.log; }
free_gb() { awk '/MemFree/ {print int($2/1048576)}' /proc/meminfo; }
need() { while [ "$(free_gb)" -lt "$1" ]; do sleep 30; done; }
w() { env CSRAG_WORKER=$1 CSRAG_QDRANT_HTTP=$2 CSRAG_QDRANT_GRPC=$3 CSRAG_QDRANT_CONTAINER=$4 "${@:5}"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start $(date)" > $R/RUN_RESUME3.log

need 8
w w3 6337 6338 csrag-qdrant-3 $PY experiments/run_concurrency.py --exp B4_concurrency --tier_gb 1 --readers 4 \
    > $R/B4_concurrency.log 2>&1 &
P3=$!
log "B4 started"
sleep 120; need 8
w w4 6339 6340 csrag-qdrant-4 $PY experiments/run_benchB.py --exp B1_scale --tier_gb 1 --ablations 1 \
    --methods $ABL --recover 1 > $R/B1_scale_1_ablations.log 2>&1 &
P4=$!
log "1 GB ablations started"

# The 3 GB worker belongs to the paused shell, so wait for its three result lines and for its process to end.
while [ "$(lines $R/B1_scale/tier_3/runs_w2.jsonl)" -lt 3 ]; do sleep 60; done
# 22040 is the Windows process ID of that worker.
while powershell.exe -NoProfile -Command "if (Get-Process -Id 22040 -ErrorAction SilentlyContinue) {exit 0} else {exit 1}"; do sleep 30; done
sleep 60
log "3 GB tier finished"

need 15
w w1 6333 6334 csrag-qdrant $PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 \
    --methods proposed,full_history,doc_guard --recover 1 > $R/B1_scale_10_w1.log 2>&1 &
P1=$!
log "10 GB tier started"

wait $P3; log "B4 finished"
wait $P4; log "1 GB ablations finished"
sleep 600   # let the 10 GB worker reach its working memory before the next worker starts

need 10
w w3 6337 6338 csrag-qdrant-3 $PY experiments/run_benchB.py --exp B1_stress --tier_gb 1 --methods $ALL7 \
    --disorder '{"delay_median_s": 600}' > $R/B1_stress_1.log 2>&1
log "stress finished"
need 10
w w3 6337 6338 csrag-qdrant-3 $PY experiments/run_benchB.py --exp B1_seed1 --tier_gb 1 --methods $ALL7 --seed 1 \
    > $R/B1_seed1_1.log 2>&1
log "seed 1 finished"
need 10
w w3 6337 6338 csrag-qdrant-3 $PY experiments/run_benchB.py --exp B1_seed2 --tier_gb 1 --methods $ALL7 --seed 2 \
    > $R/B1_seed2_1.log 2>&1
log "seed 2 finished"

wait $P1; log "10 GB tier finished"
log "end"
