#!/usr/bin/env bash
# Second resume after the host ran short of memory again (a game used about 10.7 GB next to the runs).
# Done and kept: A1, readers, A5, A6, B2, B2 gate, B1 0.25 GB (all policies), B1 1 GB (7 main policies).
# Remaining, in memory-safe phases, one Qdrant container per worker (none of these report throughput):
#   phase 1  B5 exposure analysis (publication rules only)
#   phase 2  1 GB ablations + whole revisions | 3 GB tier | B4 concurrency (1 GB)
#   phase 3  10 GB tier (two workers)
#   phase 4  stress setting and two extra delivery seeds on the 1 GB tier (three workers)
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
R=results_v2
ALL7=overwrite,chunk_lww,incremental,doc_guard,tx_travel,full_history,proposed
ABL=full_history_per_revision,proposed_no_holdback,proposed_no_revert,proposed_inplace_commit,proposed_risk_gated
log() { echo "$* $(date)" >> $R/RUN_RESUME2.log; }
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start $(date)" > $R/RUN_RESUME2.log
w() { env CSRAG_WORKER=$1 CSRAG_QDRANT_HTTP=$2 CSRAG_QDRANT_GRPC=$3 CSRAG_QDRANT_CONTAINER=$4 "${@:5}"; }

$PY experiments/run_exposure.py --exp B5_exposure --seeds 0,1,2 > $R/B5_exposure.log 2>&1
log "phase 1 (B5) finished"

w w1 6333 6334 csrag-qdrant   $PY experiments/run_benchB.py --exp B1_scale --tier_gb 1 --ablations 1 \
    --methods $ABL --recover 1 > $R/B1_scale_1_ablations.log 2>&1 &
P1=$!
w w2 6335 6336 csrag-qdrant-2 $PY experiments/run_benchB.py --exp B1_scale --tier_gb 3 \
    --methods doc_guard,full_history,proposed,full_history_per_revision --recover 1 > $R/B1_scale_3.log 2>&1 &
P2=$!
w w3 6337 6338 csrag-qdrant-3 $PY experiments/run_concurrency.py --exp B4_concurrency --tier_gb 1 --readers 4 \
    > $R/B4_concurrency.log 2>&1 &
P3=$!
wait $P1 $P2 $P3
log "phase 2 (1 GB ablations, 3 GB, B4) finished"

w w1 6333 6334 csrag-qdrant   $PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 \
    --methods doc_guard,proposed --recover 1 > $R/B1_scale_10_w1.log 2>&1 &
P1=$!
w w2 6335 6336 csrag-qdrant-2 $PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 \
    --methods full_history --recover 1 > $R/B1_scale_10_w2.log 2>&1 &
P2=$!
wait $P1 $P2
log "phase 3 (10 GB) finished"

w w1 6333 6334 csrag-qdrant   $PY experiments/run_benchB.py --exp B1_stress --tier_gb 1 --methods $ALL7 \
    --disorder '{"delay_median_s": 600}' > $R/B1_stress_1.log 2>&1 &
P1=$!
w w2 6335 6336 csrag-qdrant-2 $PY experiments/run_benchB.py --exp B1_seed1 --tier_gb 1 --methods $ALL7 --seed 1 \
    > $R/B1_seed1_1.log 2>&1 &
P2=$!
w w3 6337 6338 csrag-qdrant-3 $PY experiments/run_benchB.py --exp B1_seed2 --tier_gb 1 --methods $ALL7 --seed 2 \
    > $R/B1_seed2_1.log 2>&1 &
P3=$!
wait $P1 $P2 $P3
log "end"
