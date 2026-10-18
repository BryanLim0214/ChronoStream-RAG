#!/usr/bin/env bash
# Sixth schedule (2026-10-18, about 00:05; the user chose "one at a time"). Windows commit (RAM plus a
# 16 GB page file) is the limit, so at most two runs go at once, each behind a commit gate:
#   lane A (10 GB): wait for the running worker (ChronoStream-RAG, Full history), then Document guard
#   lane B (1 GB):  wait for the running B4 worker, then the stress setting, seed 1 and seed 2, in turn
# None of these runs reports throughput.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
R=results_v2
ALL7=overwrite,chunk_lww,incremental,doc_guard,tx_travel,full_history,proposed
log() { echo "$* $(date)" >> $R/RUN_RESUME6.log; }
free_gb() { awk '/MemFree/ {print int($2/1048576)}' /proc/meminfo; }
commit_gb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" | tr -d '\r'; }
need() { while [ "$(free_gb)" -lt "$1" ] || [ "$(commit_gb)" -lt "$2" ]; do sleep 30; done; }
alive() { powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }) {exit 0} else {exit 1}"; }
w() { env CSRAG_WORKER=$1 CSRAG_QDRANT_HTTP=$2 CSRAG_QDRANT_GRPC=$3 CSRAG_QDRANT_CONTAINER=$4 "${@:5}"; }
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start $(date)" > $R/RUN_RESUME6.log

( while alive 'tier_gb 10'; do sleep 60; done; log "10 GB w1 finished"
  need 10 18; log "10 GB w5 (doc_guard) started"
  w w5 6339 6340 csrag-qdrant-4 $PY experiments/run_benchB.py --exp B1_scale --tier_gb 10 \
      --methods doc_guard --recover 1 > $R/B1_scale_10_w5.log 2>&1
  log "10 GB w5 finished" ) &
LA=$!
( while alive 'run_concurrency'; do sleep 60; done; log "B4 finished"
  for job in "B1_stress --disorder {\"delay_median_s\":600}" "B1_seed1 --seed 1" "B1_seed2 --seed 2"; do
    set -- $job; exp=$1; shift
    need 6 10; log "$exp started"
    w w2 6335 6336 csrag-qdrant-2 $PY experiments/run_benchB.py --exp $exp --tier_gb 1 --methods $ALL7 "$@" \
        > $R/${exp}_1.log 2>&1
    log "$exp finished"
  done ) &
LB=$!
wait $LA $LB
log "end"
