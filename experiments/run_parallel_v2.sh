#!/usr/bin/env bash
# Faster schedule for the steps after the 1 GB tier (replaces the sequential tail of run_resume_v2.sh).
# The 1 GB tier keeps running alone, so its throughput comparisons are measured on a quiet machine.
# The 3 and 10 GB tiers measure the cost of keeping history (stored vectors, Qdrant memory) and run
# in parallel, one Qdrant container per worker; their throughput is NOT comparable and is not reported.
# Policies at 3/10 GB: only those in the scale figure. Commit-time travel runs only up to 1 GB and
# whole-revision history up to 3 GB: both store vectors for every revision, and running them on the
# larger tiers next to other workers would exceed the 23.4 GB Docker memory limit.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
R=results_v2
log() { echo "$* $(date)" >> $R/RUN_PARALLEL.log; }
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start $(date)" > $R/RUN_PARALLEL.log
# worker: name http grpc container command...
w() { env CSRAG_WORKER=$1 CSRAG_QDRANT_HTTP=$2 CSRAG_QDRANT_GRPC=$3 CSRAG_QDRANT_CONTAINER=$4 "${@:5}"; }

# ---- wait for the sequential 1 GB tier (it writes provenance.json when it ends)
while [ ! -f $R/B1_scale/tier_1/provenance.json ]; do sleep 30; done
log "1 GB tier finished"

# ---- phase 1: 3 GB tier on two servers, B4 concurrency on a third
w w1 6333 6334 csrag-qdrant   $PY experiments/run_benchB.py --exp B1_scale --tier_gb 3 \
    --methods full_history_per_revision --recover 1 > $R/B1_scale_3_w1.log 2>&1 &
P1=$!
w w2 6335 6336 csrag-qdrant-2 $PY experiments/run_benchB.py --exp B1_scale --tier_gb 3 \
    --methods doc_guard,full_history,proposed --recover 1 > $R/B1_scale_3_w2.log 2>&1 &
P2=$!
w w3 6337 6338 csrag-qdrant-3 $PY experiments/run_concurrency.py --exp B4_concurrency --tier_gb 1 --readers 4 \
    > $R/B4_concurrency.log 2>&1 &
P3=$!
wait $P1 $P2 $P3
log "3 GB tier and B4 finished"

# ---- phase 2: 10 GB tier on two servers
w w1 6333 6334 csrag-qdrant   $PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 \
    --methods doc_guard,proposed --recover 1 > $R/B1_scale_10_w1.log 2>&1 &
P1=$!
w w2 6335 6336 csrag-qdrant-2 $PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 \
    --methods full_history --recover 1 > $R/B1_scale_10_w2.log 2>&1 &
P2=$!
wait $P1 $P2
log "end"
