#!/usr/bin/env bash
# Follow-up runs requested by review R5 (all on top of run_parallel_v2.sh, never at the same time as
# the sequential 1 GB tier):
#   B5  exposure time per later-undone revision, by revert speed and editor type, with detector labels and
#       MediaWiki tags, plus a supersede-only hold (publication rules only, no index); starts when the
#       1 GB tier has finished.
#   B1_stress  pre-registered stress setting (median extra delay 10 min) on the 1 GB tier, 7 main policies.
#   B1_seed1, B1_seed2  two more delivery seeds on the 1 GB tier, 7 main policies.
#   The three 1 GB runs start after run_parallel_v2.sh has ended and run in parallel on separate Qdrant
#   containers; they report index states, not throughput.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
R=results_v2
ALL7=overwrite,chunk_lww,incremental,doc_guard,tx_travel,full_history,proposed
log() { echo "$* $(date)" >> $R/RUN_FOLLOWUP.log; }
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start $(date)" > $R/RUN_FOLLOWUP.log
w() { env CSRAG_WORKER=$1 CSRAG_QDRANT_HTTP=$2 CSRAG_QDRANT_GRPC=$3 CSRAG_QDRANT_CONTAINER=$4 "${@:5}"; }

while [ ! -f $R/B1_scale/tier_1/provenance.json ]; do sleep 30; done
$PY experiments/run_exposure.py --exp B5_exposure --seeds 0,1,2 > $R/B5_exposure.log 2>&1
log "B5 exposure finished"

while ! grep -q "^end" $R/RUN_PARALLEL.log 2>/dev/null; do sleep 30; done
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
