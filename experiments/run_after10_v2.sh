#!/usr/bin/env bash
# Queued by the user's choice (2026-10-18, 01:40): after the 10 GB runs, run the test suite on the
# snapshot-commit fix, then rerun B4 for ChronoStream-RAG with snapshot commits (its earlier record had the bug).
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
R=results_v2
log() { echo "$* $(date)" >> $R/RUN_AFTER10.log; }
alive() { powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }) {exit 0} else {exit 1}"; }
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start $(date)" > $R/RUN_AFTER10.log
while alive 'tier_gb 10'; do sleep 60; done
log "10 GB runs finished; tests started"
env CSRAG_QDRANT_HTTP=6337 CSRAG_QDRANT_GRPC=6338 CSRAG_QDRANT_CONTAINER=csrag-qdrant-3 $PY -m pytest -q tests > $R/TESTS_after_fix.log 2>&1
log "tests finished: $(tail -1 $R/TESTS_after_fix.log)"
env CSRAG_WORKER=w3 CSRAG_QDRANT_HTTP=6337 CSRAG_QDRANT_GRPC=6338 CSRAG_QDRANT_CONTAINER=csrag-qdrant-3 \
    $PY experiments/run_concurrency.py --exp B4_concurrency --tier_gb 1 --readers 4 --methods proposed \
    > $R/B4_concurrency_proposed.log 2>&1
log "B4 proposed finished"
