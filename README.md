# ChronoStream-RAG

Code and data for the paper *Keeping Retrieval Indexes Consistent When Document Updates Arrive Out of Order: Measurements on Wikipedia Edit Histories* (Lynette Osako, Panhapiseth Lim, Rajan Yadav; University of Texas Permian Basin).

A retrieval-augmented generation (RAG) system stores each document as many chunks in a vector index. When revisions of a document arrive late, twice, or out of order, the index can show:
- **torn** pages, which mix chunks of two revisions;
- **stale** pages, which show an older revision;
- **exposed** pages, which show an edit that a later revert undoes.

This repository replays real Wikipedia revision histories through a Qdrant vector database under simulated delivery delays. It then checks the index state at chosen times against the true history, without any language model.

## What is here
| Path | Contents |
|---|---|
| `csrag/` | The update policies (`methods.py`), the content-hash revert detector (`revert.py`), the discrete-event replay (`replay.py`, `stream.py`), the index-state and attribution measures (`metrics.py`), the Qdrant wrapper (`store.py`), the Parquet archive (`cold.py`), the answer models (`readers.py`) and statistics over documents (`stats.py`). |
| `experiments/` | One runner per experiment. `run_all_v2.sh` runs every experiment in the paper, in order. `gen_paper_data.py` writes every number and figure data file the paper prints. |
| `configs/` | `primary_metrics.yaml`, the pre-registration (defaults, hypotheses and statistics, committed before the main runs); `benchB_fetch.yaml`, the Wikipedia fetch rules. |
| `data/benchA/` | The 20 hand-written fact timelines of the question set (semi-synthetic). |
| `tests/` | Unit tests for revert detection, attribution and data-leakage guards, plus randomized tests of the history store against a live Qdrant server. |
| `results_v2/` | Where the runs write their result files (each run folder has a `provenance.json` with its commit and settings). The folder is not tracked by git; the result files of the reported runs are archived separately. |
| `legacy/` | The original release. It is not used by this code; `legacy/README.md` explains why its numbers are not reported. |

## Reproducing
1. Python 3.12 with the pinned packages in `requirements.lock.txt` (PyTorch 2.5.1 with CUDA 12.1).
2. A Qdrant 1.19.2 server in Docker on ports 6333/6334 (`docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.19.2`). Our runs used the image `qdrant/qdrant:latest`, which reported version 1.19.2 (commit 016542a).
3. Models: `python -m csrag.models_download`. The Wikipedia histories: `python -m csrag.wiki_fetch` (rules in `configs/benchB_fetch.yaml`). Large files go to `../cache` (set `CSRAG_CACHE` to change this).
4. `bash experiments/run_all_v2.sh`, then `python experiments/gen_paper_data.py --paper ../paper`.

Delivery delays and duplicate messages are simulated (`csrag/stream.py`); the revision texts, edit times, editor account types and revert tags are real. All timing and memory figures are measured on one machine. Nothing is extrapolated.
