#!/usr/bin/env bash
# Fifth schedule (2026-10-17, about 23:00). At 22:37 the shell of run_resume3_v2.sh, which the memory guard
# had not stopped after all, and run_resume4_v2.sh both started a 10 GB worker on csrag-qdrant under the same
# worker name. Both ran out of Windows commit (RAM plus page file, 63.9 GB) and Docker Desktop then crashed.
# Everything from that clash is set aside in results_v2/_aborted_2026-10-17_2253. This is now the only
# schedule. Workers start only when enough commit (not only RAM) is free. One chain per Qdrant container:
#   csrag-qdrant    10 GB tier: ChronoStream-RAG, Full history
#   csrag-qdrant-3  B4 (the two policies left; the first two are kept from the earlier run), then seed 2
#   csrag-qdrant-2  stress setting
#   csrag-qdrant-4  seed 1, then the 10 GB tier: Document guard (worker w5)
# None of these runs reports throughput.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
R=results_v2
ALL7=overwrite,chunk_lww,incremental,doc_guard,tx_travel,full_history,proposed
log() { echo "$* $(date)" >> $R/RUN_RESUME5.log; }
free_gb() { awk '/MemFree/ {print int($2/1048576)}' /proc/meminfo; }
commit_gb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" | tr -d '\r'; }
need() { while [ "$(free_gb)" -lt "$1" ] || [ "$(commit_gb)" -lt "$2" ]; do sleep 30; done; }
w() { env CSRAG_WORKER=$1 CSRAG_QDRANT_HTTP=$2 CSRAG_QDRANT_GRPC=$3 CSRAG_QDRANT_CONTAINER=$4 "${@:5}"; }
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start $(date)" > $R/RUN_RESUME5.log

need 14 24
w w1 6333 6334 csrag-qdrant $PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 \
    --methods proposed,full_history --recover 1 > $R/B1_scale_10_w1.log 2>&1 &
P1=$!; log "10 GB w1 (proposed, full_history) started"
sleep 600   # let the 10 GB worker load its tier before the others start

( need 6 12; log "B4 rest started"
  w w3 6337 6338 csrag-qdrant-3 $PY experiments/run_concurrency.py --exp B4_concurrency --tier_gb 1 --readers 4 \
      --methods proposed_inplace_commit,proposed > $R/B4_concurrency_rest.log 2>&1
  log "B4 rest finished"; need 6 12; log "seed 2 started"
  w w3 6337 6338 csrag-qdrant-3 $PY experiments/run_benchB.py --exp B1_seed2 --tier_gb 1 --methods $ALL7 --seed 2 \
      > $R/B1_seed2_1.log 2>&1
  log "seed 2 finished" ) &
CA=$!
sleep 120
( need 6 12; log "stress started"
  w w2 6335 6336 csrag-qdrant-2 $PY experiments/run_benchB.py --exp B1_stress --tier_gb 1 --methods $ALL7 \
      --disorder '{"delay_median_s": 600}' > $R/B1_stress_1.log 2>&1
  log "stress finished" ) &
CB=$!
sleep 120
( need 6 12; log "seed 1 started"
  w w4 6339 6340 csrag-qdrant-4 $PY experiments/run_benchB.py --exp B1_seed1 --tier_gb 1 --methods $ALL7 --seed 1 \
      > $R/B1_seed1_1.log 2>&1
  log "seed 1 finished"; need 14 24; log "10 GB w5 (doc_guard) started"
  w w5 6339 6340 csrag-qdrant-4 $PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 \
      --methods doc_guard --recover 1 > $R/B1_scale_10_w5.log 2>&1
  log "10 GB w5 finished" ) &
CC=$!

wait $P1; log "10 GB w1 finished"
wait $CA $CB $CC
log "end"
