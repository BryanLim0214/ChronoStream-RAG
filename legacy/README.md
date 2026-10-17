# Legacy material from the original release (commit 3d2b8d0)

Nothing in this folder is used by the measured v2 harness (`csrag/`, `experiments/`) or by the paper.
It is kept for the record. `audit/CLAIM_INVENTORY.md` in the paper workspace explains, claim by claim,
why the original numbers are not reported: modeled latency and memory, a 20-topic corpus copied three
times, a revert detector that read hand-written labels, extrapolated scale results, and an in-memory
vector store. `experiments/make_figures.py` and `experiments/run_B_all.sh` are superseded v2 drafts
(the paper's figures are TikZ/pgfplots files; every experiment runs from `experiments/run_all_v2.sh`).
