# Code changes and deviations from the pre-registration

`repo/configs/primary_metrics.yaml` (commit `1e2e394`, 2026-10-17 15:05 -0500) says: "changes require a new file and a note in audit/CODE_CHANGES.md". This file is that note.

Every number in the paper comes from one run of `experiments/run_all_v2.sh`. The run started 2026-10-17 17:03 at commit **`40dcd34`** from a clean tree (`RUN_ALL.log`: `dirty=0`).
- A1 and its readers ran at `40dcd34`.
- All later steps ran at **`a7a35d3`**, which changes only the attribution rule (see below).
- Each step's `provenance.json` records its commit and whether the tree was clean. Earlier results are kept under `results_v2/_superseded/` and `results_v2/_aborted_B1_scale_pre_review/`, and the paper does not use them.

## 1. Fixes to the original release (commit `3d2b8d0`)
`audit/CLAIM_INVENTORY.md` lists each claim of the original release. It gives what the code actually did and whether the new paper keeps, re-measures or drops the claim. In short:

**Data**
- The data are real Wikipedia histories, not hand-written revisions with real comments attached to them.
- There are 20 timelines, not 20 topics copied three times.

**Revert handling**
- A content-hash revert detector replaces the string matcher and the hand-written labels.
- The holdback is now a real fixed-lag watermark. In the original code, `watermark_window_sec` was never read.

**Measurement**
- Memory and latency are measured, not computed from formulas.
- The 1–10 GB figures were extrapolated. They are replaced by tiers that were actually run.
- Seed-level p-values are replaced by statistics over documents.

## 2. Changes after the pre-registration commit
| Commit | Time (-0500) | Change | Effect on results |
|---|---|---|---|
| `693b308` | 15:05 | Benchmark A runner: ablation and sweep options. | None (options only). |
| `58a1fc8` | 15:22 | **Interval layout** for valid-time history (one stored chunk per unbroken run of revisions). Asynchronous writes with a barrier before reads. Read retries. Ablation configs. | This defines the "Full history" and ChronoStream-RAG policies that the paper measures. The per-revision layout remains as the ablation "Full history (whole revisions)". |
| `fc841d1` | 15:40 | Readers, cluster statistics, holdback replay, Benchmark B and concurrency runners. Barrier before publishing a snapshot. | New experiments; no change to policies. |
| `2806bc5` | 15:53 | SHA-1 verification script. | None. |
| `188305d` | 16:39 | Snapshot-commit barrier made conditional on concurrent readers. | **Reverted in `40dcd34`** (R3 finding M11): the barrier cost now always counts in throughput. |
| `40dcd34` | 16:59 | Fixes from the code-against-claims review (R3). See below. | Every reported number comes from this commit. |

### Fixes in `40dcd34` (from `audit/REVIEW_R3_code_claims.md`)
1. **Attribution rule** (`csrag/metrics.py`, `attribute`).
   - Before, a probe that was wrong even with respect to revisions already ready could be explained away by the clocks ("not yet arrived" or "in flight").
   - Now such a probe is "held by policy" if the policy is holding the revision it should show, and an "index error" otherwise. The clocks explain only the gap between the best ready revision and the latest correct revision. (R3 H2.)
2. **Snapshot commit always waits for its writes** (`HistoryMethod.concurrent_readers = True` by default). Throughput of snapshot-commit policies now includes one synchronous round trip per commit. (R3 M11.)
3. **Re-serving revisions that turn out not to be reverted** (`HistoryMethod._restore`).
   - Before, a revision that was suppressed or withdrawn stayed hidden if a revision arriving later changed the revert detection result.
   - Now it is served again. Counters: `unreverted_restored`, `unreverted_lost`.
4. **Holdback simulator gate** (`experiments/run_holdback.py --gate_tier`).
   - The holdback sweep replays the publication rules without an index (`csrag/holdback_sim.py`).
   - On the 0.25 GB tier, its head-change log must equal the full system's for every history policy run on that tier. The run asserts this, and the result is in `results_v2/B2_holdback/gate_tier0.25.json`. (R3 M1.)
5. **Benchmark B runner.**
   - Point counts are exact (no barrier sentinel).
   - Head logs and per-policy probe files are saved.
   - Each policy starts from a restarted Qdrant container.

### Fix in `a7a35d3` (found while checking the new A1 results, before any Benchmark B step started)
**Attribution of exposed states** (`csrag/metrics.py`, `_exposed_cause`).
- **The bug.** The rule checked the clocks of the latest good revision. So a vandal edit shown *before its revert was even made* was counted as an "index error". Example: A1, document A02. The edit was made at 66 s, the probe ran at 70.3 s, and the revert was made at 71 s. The rule in Section III-C of the paper says this case is explained by the clocks.
- **The new rule.** An exposed state uses the clocks of the revert that undoes the shown edit:
  - "not yet arrived" if the revert had not been made or had not arrived;
  - "in flight" while it is being embedded;
  - "held by policy" if the policy is holding it;
  - "index error" only once it is ready and still not applied.
- **Test.** `test_attribution_of_exposed_state_uses_the_revert_clocks` covers each case.
- **Scope.**
  - A1 ran at `40dcd34`. Its `attr` column has the old rule for 2 of 1,990 Document guard probes and the corresponding probes of the other history policies. The paper does not report Benchmark A attribution.
  - Every step from A5 on, and all of Benchmark B, ran at `a7a35d3`. Each run's `provenance.json` records the commit it ran at.
- **Other changes in this commit.**
  - `gen_paper_data.py` changes listed in section 3.
  - State shares now exclude probes taken before any good revision of the document was ready (`cls == "no_target"`; 0.6% of Benchmark A current probes).

## 3. Changes to analysis only (after `40dcd34`, no effect on any run)
`experiments/gen_paper_data.py` was rewritten after the R3 review. It reads result files only.
- Table columns:
  - A Missing column was added.
  - The ablation "without snapshot commit" is now printed.
  - The ablation label reads "without revert detection", because the switch disables both suppression and retraction.
- Probe filtering: current probes with no target are excluded (`attr == "no_target"`).
- Reader accuracy by context state:
  - It is now a mean of per-document means with a bootstrap interval. It was previously a pooled mean.
  - It is computed over the seven main policies only, so ablation rows do not count twice.
- Constants are computed from data files instead of typed in: the legacy-article count (42 found of 45 titles), the revision and query counts, the seed count, the maximum revisions per article and the delay median.
- The pre-registered Wilcoxon signed-rank tests with Holm correction and the equivalence check are computed and written to `audit/STATS_TESTS.md`.
- Benchmark B states are reported separately for:
  - random-time probes;
  - probes placed just after an undone edit becomes ready;
  - final probes;
  - as-of probes.

  Before, these were mixed together.

## 3b. Machine state during the runs
- Benchmark A (A1, readers, A5, A6) and B2 (no index): an idle game server with its database and an idle Supabase stack ran in Docker on the same machine. Together they used about 17% of one core and 6.5 GB of the 23.4 GB Docker memory (`docker stats`, 2026-10-17 18:1x). None of these steps reports timing that the paper interprets.
- B1, 0.25 GB tier:
  - The Supabase stack was stopped at about 18:17, during the first policy (Slot overwrite).
  - The game server and its database were stopped by the user between about 18:20 and 18:26, while the first four policies ran (Slot overwrite, Slot latest-wins, Incremental indexing, Document guard).
- From the 0.25 GB tier's fifth policy on, only the experiment's Qdrant container ran in Docker.
- Docker's memory is capped at 23.4 GB by the WSL configuration, which was left unchanged.

### Host memory shortage and resume (2026-10-17, about 19:00)
- **What happened.** During the 1 GB tier, after 3 of 12 policies, the host ran out of memory: 2.9 GB free of 47.9 GB. Docker's WSL VM (`vmmem`) held 18.7 GB, most of it Linux page cache. Claude Code then stopped the pipeline's outer shell.
- **Cleanup.** The orphaned `run_all_v2.sh` had already moved on to B4. The user approved a clean restart, and then:
  - every experiment process was stopped;
  - Docker Desktop was stopped, WSL was shut down (`wsl --shutdown`, with only the `docker-desktop` distribution present), and Docker was started again. After the restart, 22.4 GB was available.
- **Kept.** Everything that finished before the stop:
  - A1, readers, A5, A6;
  - B2 and the B2 gate;
  - the full B1 0.25 GB tier.
- **Set aside, not used.** The partial 1 GB tier and partial B4 were moved to `results_v2/_aborted_oom_2026-10-17/`.
- **Code change** (commit `5e91930`). `restart_qdrant` now drops the VM page cache while Qdrant is stopped between policies (`echo 3 > /proc/sys/vm/drop_caches` in a throwaway privileged container). No stored data changes. Each policy already started with a freshly restarted Qdrant, so the only difference is that the OS cache of earlier policies no longer carries over.
  - The 0.25 GB tier ran without this cache drop.
  - The 1, 3 and 10 GB tiers and B4 run with it.
- **Resume.** The remaining steps first ran from `experiments/run_resume_v2.sh` at `5e91930`, with the same options as `run_all_v2.sh`, logging to `RUN_RESUME.log`. Only its 1 GB tier was used (see below).

### Other load during the 1 GB tier
- A game started on the host at 20:05:18 and used about 10.6 GB of memory and an unknown share of CPU and GPU.
  - The seven main policies of the 1 GB tier had all finished by 20:04:25, so their timings are clean.
  - Whole-revision Full history and the ablations of the 1 GB tier ran after that point.
- The throughput comparison with and without snapshot commit is therefore taken from the 0.25 GB tier, where both ran on a quiet machine. The paper says so; the 1 GB pair is used only if the game had ended before both ran.

### Second memory shortage and stop (2026-10-17, about 20:20–20:26)
- **What happened.** With the game using 10.7 GB and Docker's VM 13.2 GB, available memory fell to 0.63 GB. Claude Code stopped the waiting orchestrator shells. The lead then:
  - stopped the orphaned orchestrators;
  - stopped the 1 GB run during `full_history_per_revision`;
  - stopped the Qdrant container, to protect the host.
- **Kept.** The seven main policies of the 1 GB tier (`B1_scale/tier_1`). `provenance_note.json` replaces the end-of-tier provenance file.
- **Not run on the 1 GB tier.** Whole revisions and the ablations.
- **Next.** The remaining steps are in `experiments/run_resume2_v2.sh` (commit `373b90f`). They start only when the user asks.
- **Restart.** The user asked to resume at about 21:10.
  - WSL was shut down and Docker restarted, which brought available memory to 25.5 GB.
  - The script ran at `7f6b93d`, with a memory gate before each worker.
- **Final schedule:**
  1. B5.
  2. The 1 GB ablations and whole revisions, in parallel with the 3 GB tier (Document guard, Full history, \sys).
  3. B4.
  4. The 10 GB tier (two workers if at least 22 GB is free, else one).
  5. The stress setting and two seeds on the 1 GB tier.
- **What does not run.** Whole revisions do not run at 3 or 10 GB, and Commit-time travel does not run above 1 GB.

### Faster schedule for the larger tiers (user request, 2026-10-17 about 19:20)
The sequential schedule would have taken about 10 more hours. At the user's request:
- **1 GB tier.** It keeps running alone, unchanged (`run_resume_v2.sh`, commit `5e91930`), so its throughput comparisons (for example, snapshot commit on and off) are measured on a quiet machine. The resume script's shell was stopped after this tier started, so it never ran its later steps.
- **`experiments/run_parallel_v2.sh`** (commit `91c5583`, worker support in `50c1144`) waits for the 1 GB tier, then:
  - **3 GB tier:** whole-revision Full history on one Qdrant container; Document guard, Full history and \sys on a second; and B4 (1 GB) on a third, all in parallel.
  - **10 GB tier:** Document guard and \sys on one container and Full history on another, in parallel.
- **Separate servers.** Each worker uses its own Qdrant container (`csrag-qdrant`, `csrag-qdrant-2`, `csrag-qdrant-3`, same image digest) and its own run and provenance files (`runs_w1.jsonl` …).
- **What changes for the results.**
  - Index states, stored vectors and each container's own memory metric do not depend on other workers.
  - Throughput and recovery time at 3 and 10 GB do, so the paper does not report them.
  - Slot overwrite, Slot latest-wins and Incremental indexing run only at 0.25 and 1 GB, because no claim about cost at scale concerns them.
  - Commit-time travel runs only at 0.25 and 1 GB, and whole-revision Full history only up to 3 GB. Both store vectors for every revision, and running them on the larger tiers next to other workers would exceed the 23.4 GB Docker memory limit. The paper states this; nothing is extrapolated.

### Third memory shortage and third schedule (2026-10-17, about 21:30–21:50)
- **What happened.** The 1 GB ablation worker of `run_resume2_v2.sh` failed at start: its gRPC port was not ready (fixed in `restart_qdrant`, commit `d40c1e7`). Its relaunch on `csrag-qdrant-4` brought available memory down to 1.7 GB, so the lead stopped it (no results kept; its log is `B1_scale_1_ablations_stopped.log`). The 3 GB worker kept running.
- **Memory reclaim.** Docker's VM held 17.2 GB in Windows although 22 GB was free inside it. From 21:44 a loop runs every 90 s (every 30 s from about 21:57, after the page cache refilled within a minute) in a throwaway privileged container: `sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory`. Compaction lets the VM hand freed memory back to Windows (vmmem 17.2 → 12.6 GB in four rounds; available 6.9 → 14.1 GB). No setting and no stored data changes. Qdrant re-reads from disk what was dropped, which can slow these runs; none of them reports throughput. The loop logs to `results_v2/VM_RECLAIM.log`.
- **Old schedule paused.** Its phase 4 could start two 10 GB workers, which would need about twice the Python memory. Its shell was paused (`kill -STOP`) while its 3 GB worker continued.
- **`experiments/run_resume3_v2.sh`** (commit `8e4e962`) replaces phases 3–5:
  1. B4 (1 GB) on `csrag-qdrant-3` and the 1 GB ablations on `csrag-qdrant-4`, beside the 3 GB worker.
  2. The 10 GB tier as one worker (\sys, Full history, Document guard, in that order) as soon as the 3 GB worker ends.
  3. The stress setting, then seeds 1 and 2, one after another on `csrag-qdrant-3`.
- **Dropped.** Whole-revision Full history on the 1 GB tier. It is reported on the 0.25 GB tier only, and the paper says why.
- **Leftover collections (found 21:57).** A run stopped in the middle of a policy leaves its Qdrant collection behind, and Qdrant loads it at every later start. `csrag-qdrant-4` still held the 7.8 GB whole-revision collection of the stopped 1 GB relaunch, and the stopped `csrag-qdrant` a 4.6 GB one from the 1 GB run stopped at 20:26. Together with the page cache they explain why Docker's VM kept refilling.
  - The lead deleted both (API delete on `csrag-qdrant-4`; directory removal on the stopped `csrag-qdrant`). No result comes from either collection.
  - Mapped file memory in the VM fell from 10.6 to 1.5 GB, and available host memory rose from 4.4 to 9.0 GB.
  - Earlier reported runs were not affected. Each container lists no other leftover, and the reported Qdrant memory of the 0.25, 1 and 3 GB runs is in line with their own collections.
  - The first 1 GB ablation, `proposed_no_holdback`, started with the 7.8 GB leftover loaded. Its Qdrant memory and timing are therefore not clean; the paper reports neither for ablations. Its index states do not depend on other collections.
  - **Code change** (commit `7efe8e2`). `restart_qdrant` deletes every collection on the worker's own server once it is ready, so every policy starts from an empty server. It was tested on the idle `csrag-qdrant` (a dummy collection was removed). Runs launched from then on (10 GB tier, stress setting, seeds) use it; the B4, ablation and 3 GB workers already running do not.
- **B4 ran beside other workers.** Its torn and missing counts depend on how reads and writes interleave, so other load can change them. The paper reports them as observed counts.

### Fourth memory shortage (2026-10-17, about 22:20)
- **What happened.** A game started on the host and used 9.2 GB. Available memory fell to about 3 GB, and Claude Code's memory guard stopped three idle background shells: the paused `run_resume2_v2.sh`, `run_resume3_v2.sh` and the VM reclaim loop.
- **Still running.** The worker processes those shells had started kept running: the 3 GB tier (last policy, \sys), B4 (policy 2 of 4) and the 1 GB ablations (policy 3 of 4). Their results are written by the workers themselves.
- **Not started.** The 10 GB tier, the stress setting and seeds 1 and 2. Per the guard's instruction, nothing is restarted until the user asks.
- **Timing.** The game shares CPU and GPU with these workers. None of them reports throughput.
- **User decision (about 22:30).** Keep the running workers; start nothing new, including the reclaim loop, until the user says so.
- **Fourth schedule (about 22:40).** The game had closed (18.4 GB available) and the user asked to go faster. `experiments/run_resume4_v2.sh` (commit `2e7472b`) splits the 10 GB tier over two workers (ChronoStream-RAG and Full history on `csrag-qdrant`; Document guard later on `csrag-qdrant-4` as worker w5) and runs the stress setting and seeds 1 and 2 beside it, each behind a memory gate. The reclaim loop runs again.

### Duplicate 10 GB workers and Docker crash (2026-10-17, 22:37-22:55)
- **What happened.** The memory guard had stopped only the wrapper shells. The `run_resume3_v2.sh` and `run_resume2_v2.sh` shells kept running, and at 22:37 `run_resume3_v2.sh` started a 10 GB worker on `csrag-qdrant` as w1, one second before `run_resume4_v2.sh` started its own w1 on the same container, writing the same log and run files. Both ran out of Windows commit (RAM plus page file, 63.9 GB): `MemoryError`, `can't start new thread`. Docker Desktop then crashed (WSL bootstrap exit 0xc00000fd).
- **Set aside.** The 10 GB files, the partial stress and seed 1 runs and their logs are in `results_v2/_aborted_2026-10-17_2253/`. None is used.
- **Kept.** The first two B4 policies (incremental, Document guard), whose records the B4 worker had printed before Docker stopped (`B4_concurrency.log`, copied as `B4_concurrency_first2.log`). They seed `B4_concurrency/concurrency_tier1.jsonl`; the other two policies rerun with the new `--methods` option, which appends.
- **Restart.** Every experiment process and stale shell was stopped; WSL was shut down and Docker restarted. `experiments/run_resume5_v2.sh` (commit `7b428ed`) is now the only schedule. Its gates also wait for free commit, not only free RAM.

### Commit exhausted again (2026-10-18, about 00:00)
- **What happened.** With the 10 GB worker (11.8 GB committed), B4 and the stress and seed 1 runs (about 4.6 GB each) and Docker's VM (12.3 GB committed), free Windows commit fell to 0.6 GB. The memory guard stopped the wrapper shells of `run_resume5_v2.sh` and of the reclaim loop.
- **Action.** The lead stopped the schedule's surviving inner shells, so that nothing new starts, and stopped the stress and seed 1 runs, which had finished one policy each and must restart from scratch anyway. Their partial files are not used. Free commit rose to 10.2 GB.
- **Kept running.** The 10 GB worker (ChronoStream-RAG done; Full history at 80,000 of 121,751 deliveries) and B4 (its last policy).
- **Not started yet.** The stress setting, seeds 1 and 2, and Document guard at 10 GB. They wait for the user.

### Bug found by B4: snapshot commit lost restored points (2026-10-18, about 00:45)
- **Symptom.** In B4, ChronoStream-RAG with snapshot commits showed a torn page in 1,766,600 of 3,629,912 reads (49%), against 0.8% without snapshot commits.
- **Cause** (`csrag/methods.py`). Point ids were derived from (document, chunk, start, end) only. With concurrent readers, a closed point stays until no reader needs it (vacuum). When an interval came back with the same key (a revert withdrawn, so an earlier interval is restored), its insert overwrote the closed point, and the later vacuum deleted that id, which was now the live point. The chunk then stayed missing until the document changed again.
- **Why only B4.** In every single-threaded replay no reader holds an old snapshot, so a closed point is deleted at the same commit, before any re-insert. All Benchmark A and B results, ablations and size tiers are therefore unaffected (same live points, same reads); B4 is the only experiment that registers readers.
- **Fix.** Under snapshot commits the commit number is part of every point id (`HistoryMethod._pid`), and points are closed by the id that was recorded when they were written (intervals and, for the revision layout, `rev_pids`). In-place commits keep the old ids.
- **Tests.** `test_restored_point_survives_delayed_delete` (new) fails on the old code ({a} instead of {a, b}) and passes on the fix, for both layouts; `test_snapshot_commit_with_lagging_readers` (new) passes on both.
- **Test run after the fix** (2026-10-18 12:30, commit `2f191f6`): 21 passed (`results_v2/TESTS_after_fix.log`).
- **Results.** The B4 record of ChronoStream-RAG with the bug is kept in `results_v2/_aborted_2026-10-18_0000/B4_proposed_with_id_bug.jsonl` and not used; that policy reruns on the fixed code. The other three B4 records do not use delayed deletes and are kept. The stress setting, which had started at 00:31 on the old code, is single-threaded and unaffected.

### Pause and finish (2026-10-18)
- The overnight schedule (`run_resume6_v2.sh`) finished at 04:25: 10 GB tier (all three policies), stress setting and seeds 1 and 2 (all seven policies each).
- At about 12:20 the user paused everything to restart the computer, before the queued tests and the B4 rerun had begun. After the restart, both ran in parallel on separate containers (tests on `csrag-qdrant-3`, B4 ChronoStream-RAG on `csrag-qdrant-4`), at commit `2f191f6`.

## 4. Deviations from `configs/primary_metrics.yaml`
| Pre-registered | What was done | Why |
|---|---|---|
| Benchmark B `stress_disorder` (10-min median delay) | Run on the 1 GB tier for the seven main policies (`B1_stress`), after the size tiers, with seeds 1 and 2 of the default setting. | Scheduled last, after the size tiers and the concurrency test. |
| Benchmark B `risk_gated_operating_point` (H = 1 h, risk-gated) | Measured in the holdback sweep on all revisions and, through the ablation set, as `proposed_risk_gated` on the tiers run with ablations. | The gate (`gate_tier0.25.json`) found identical head-change logs for all 7 history policies on the 0.25 GB tier, including H = 0, H = 10 min and risk-gated H = 1 h, with and without revert detection. |
| Tests: Wilcoxon + Holm "within each table" | Done for the planned comparisons of hypotheses H1, H2, H3 (Benchmark B) and H5 (Benchmark A). See `audit/STATS_TESTS.md`. | H4 (index size) is a cost measured once per tier, so there is no per-document test. |
| Equivalence margin 2 pp, ChronoStream-RAG vs full history | Reported in `audit/STATS_TESTS.md`. | ChronoStream-RAG holds edits by design, so a gap in current exactness is expected and is explained in the text. |
| "Qdrant server with real HNSW" (claim inventory #7) | Collections are configured for HNSW, but every read is restricted to one document. Qdrant answers these reads from the `doc_id` payload index with exact scoring, and at the sizes run it built HNSW graphs only for large segments (`index_indexed_vectors` is recorded per run). | The paper states this in Section V. |
| Benchmark A runs, server restart per run | Benchmark A runs share one server and recreate the collection for each run. Benchmark B restarts the container for each policy. | Stated in Section V. |
