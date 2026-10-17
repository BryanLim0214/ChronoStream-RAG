# ChronoStream-RAG: Complete Methodology, Formal Algorithms, Data Sources & Literature Bibliography

**Authors:**
- **Lynette Osako** (Student ID: `8000290404`)
- **Panhapiseth Lim** (Student ID: `8000265274`)
- **Rajan Yadav** (Student ID: `8000277867`)

---

## 1. End-to-End Methodology & Formal System Model

![ChronoStream-RAG End-to-End Architecture](../plots/fig12_architecture_diagram.png)

### 1.1 Bitemporal Event-Stream Accounting $(t_{\text{event}}, t_{\text{arrival}}, t_{\text{available}})$
Let $\mathcal{D} = \{D_1, D_2, \dots, D_{60}\}$ be a corpus of evolving multi-chunk documents. Each document $D_i$ undergoes a sequence of discrete revisions $v_{i,1}, v_{i,2}, \dots, v_{i,R_i}$ authored at source **event timestamps** $t_{\text{event}}(v_{i,1}) < t_{\text{event}}(v_{i,2}) < \dots < t_{\text{event}}(v_{i,R_i})$. Each revision $v_{i,r}$ is segmented into $m(v_{i,r}) \in \{4, 6\}$ text chunks $C(v_{i,r}) = \{c_{i,r,0}, \dots, c_{i,r,m-1}\}$.

Every streaming update event $u = (D_i, v_{i,r})$ is tracked across three distinct temporal coordinates:
1. **Source Event Time ($t_{\text{event}}(u)$):** When the revision was committed on Wikipedia (`timestamp` / `wiki_revid`).
2. **Broker Arrival Time ($t_{\text{arrival}}(u) = t_{\text{event}}(u) + \Delta_{\text{net}}(u)$):** When the update arrives at the partitioned Kafka broker (`src/kafka_pyspark_pipeline.py`). An *exogenous out-of-order network arrival* occurs whenever $t_{\text{event}}(u_a) < t_{\text{event}}(u_b)$ but $t_{\text{arrival}}(u_a) > t_{\text{arrival}}(u_b)$.
3. **Vector Index Availability Time ($t_{\text{available}}(u) = t_{\text{arrival}}(u) + \Delta_{\text{embed}}(u)$):** When all chunks of $u$ complete dense neural encoding (`sentence-transformers/paraphrase-MiniLM-L6-v2`, 384-d) and become queryable in Qdrant HNSW. An *endogenous compute-lag inversion* occurs even at $p_{\text{ooo}}=0\%$ whenever an earlier 6-chunk revision ($6 \times 3.43\text{ ms} = 20.58\text{ ms}$) finishes embedding *after* a subsequent 4-chunk revision ($4 \times 3.43\text{ ms} = 13.72\text{ ms}$).

### 1.2 Formal Definitions of the Three Streaming RAG Anomalies
1. **Intra-Document Version Tearing Rate ($\text{IDVTR}$ — *Semantic Split-Brain*):**
   For a top-$k$ retrieval result $R_k(q) = \{c_1, \dots, c_k\}$ targeting document $D_i$, $\text{IDVTR}$ is the probability that the retrieved context contains chunks originating from **two or more distinct revisions** of the *same* document simultaneously:
   $$\text{IDVTR} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \mathbb{I}\Big( \big|\{\text{rev}(c) \mid c \in R_k(q)\}\big| \ge 2 \Big)$$
   Under scalar per-chunk version checking (`M2a_ScalarChunkLWW`), when a revision consolidates a document from $m(v_{r-1}) = 6$ chunks down to $m(v_r) = 4$ chunks, slots `0..3` are updated to $v_r$ while **orphan tail chunks `4..5` from $v_{r-1}$ remain alive in Qdrant**, producing a **100.0% tearing rate** on consolidated documents (`25.61% ± 1.18%` overall).
2. **Endogenous Compute-Lag Inversion Rate ($\text{CLIR}$):**
   The probability that $t_{\text{arrival}}(v_{r-1}) \le t_{\text{arrival}}(v_r)$ at the broker, yet $t_{\text{available}}(v_{r-1}) > t_{\text{available}}(v_r)$ at the Qdrant vector sink due to variable chunk counts and worker batching stragglers.
3. **Held-Data Processing Error vs. Unarrived Network Lag (Perspective 1):**
   At any query time $t_q$, let $\mathcal{A}_i(t_q) = \{v_{i,r} \mid t_{\text{arrival}}(v_{i,r}) \le t_q\}$ be the set of revisions **already received by the system**. Discrepancies against an omniscient event-time oracle are partitioned into:
   - *Unarrived Network Lag:* $t_{\text{event}}(v^*) \le t_q < t_{\text{arrival}}(v^*)$ (`20.33%` mid-stream).
   - *Worker Compute-Lag In-Flight:* $t_{\text{arrival}}(v^*) \le t_q < t_{\text{available}}(v^*)$ (`28.33%` mid-stream).
   - *Held-Data Processing Error:* $t_{\text{available}}(v^*) \le t_q$, yet the index serves a stale, torn, or vandalized version (`41.33%` in `M1` $\to$ **`5.00%`** in `M4`).

---

## 2. Algorithmic Specification of All Five Consistency Protocols (`src/methods.py`)

| Protocol ID | Proposal Mapping | Vector Index Granularity | History Retention | Dedup & Vandalism Handling |
| :--- | :--- | :--- | :--- | :--- |
| **`M1_ArrivalOrder`** | Proposal Method 1 (*Arrival-Order Baseline*) | Per-chunk slot `(doc_id, chunk_idx)` | Head-only (overwrites in completion order) | None (`0.0%` compute saved, `22.61%` tearing) |
| **`M2a_ScalarChunkLWW`** | Proposal Method 2a (*Per-Chunk Version Check*) | Per-chunk slot `(doc_id, chunk_idx)` with `t_event` guard | Head-only (leaves orphan tail slots `c >= m(v_r)`) | Post-embed slot drop (`0.0%` compute saved, **`25.61%` tearing / `58.22%` orphan leakage**) |
| **`M2b_DocAtomicVersionCheck`** | Proposal Method 2b (*Document-Atomic Version Check*) | Document-atomic generation commit | Head-only (`0.00%` historical `AS OF` version recall) | Whole-document content-hash dedup (`23.95%` compute saved, `0.00%` tearing) |
| **`M3_FullBitemporal`** | Proposal Method 3 (*Full Bitemporal Document History*) | Document-atomic bitemporal intervals `[valid_from, valid_to)` | Unbounded hot Qdrant HNSW (`1,200` active vectors) | Exact revision dedup + algorithmic `A->B->A` revert splicing (`12.96%` compute saved) |
| **`M4_ChronoStream` (Ours)** | Proposal Method 4 (*Bounded Working History + Late-Update Correction*) | Document-atomic hot working window ($K=2$, `678` vectors) + Cold Snappy Parquet archive | Bounded Hot HNSW + Full Cold Parquet Bitemporal Splice | **Chunk-level SHA-256 delta cache (`81.37%` compute saved)** + **Algorithmic `A->B->A` hash-revert coalescing (`100%` vandalism resilience)** |

### Algorithm 1: `M4_ChronoStream` Ingestion, Watermark Coalescing & Bitemporal Splicing
1. **Whole-Revision Duplicate Check:** Given incoming event $u = (D_i, v_r, t_{\text{event}}, C(v_r))$, if $v_r$ or `content_hash` already exists in `doc_timeline[D_i]`, drop $u$ before embedding (`duplicates_dropped += 1`).
2. **Purely Algorithmic Vandalism & `A -> B -> A` Revert Coalescing (`detect_algorithmic_vandalism_anomaly` & `detect_algorithmic_revert_pair`):**
   - Without reading any oracle flag, inspect edit summary keywords (`revert`, `rv`, `undid`, `vandal`, `blanking`), structural character length collapse ($>45\%$ drop relative to prior revision), and **SHA-256 chunk-hash restoration (`A -> B -> A`)**: if revision $v_{r+1}$ restores $\ge 50\%$ of the exact chunk SHA-256 hashes of $v_{r-1}$ that were altered in $v_r$, mark $v_r$ as an algorithmically reverted anomaly and splice its validity interval to zero width $[t_{\text{event}}(v_r), t_{\text{event}}(v_r))$.
3. **Chunk-Level SHA-256 Delta Embedding Reuse:**
   - For each chunk $c_{r,j} \in C(v_r)$, compute $h_j = \text{SHA256}(\text{strip}(c_{r,j}))$.
   - If $h_j \in \text{chunk\_vector\_cache}$, reuse the cached 384-d float32 embedding (`reused_chunk_embeddings += 1`).
   - Otherwise, encode $c_{r,j}$ via `SentenceTransformer('paraphrase-MiniLM-L6-v2')` and insert into `chunk_vector_cache`.
4. **Bitemporal Interval Splicing & Bounded Hot Window ($K=2$) Eviction:**
   - Insert $v_r$ into `doc_timeline[D_i]` sorted by $t_{\text{event}}$.
   - Recompute contiguous non-overlapping valid-time intervals $[t_{\text{valid\_from}}(v_j), t_{\text{valid\_to}}(v_j)) = [t_{\text{event}}(v_j), t_{\text{event}}(v_{j+1}))$ for all clean revisions in `doc_timeline[D_i]`, with the latest clean revision having $t_{\text{valid\_to}} = +\infty$ and `is_current_head = True`.
   - Retain only the top $K=2$ most recent clean revisions of $D_i$ in hot Qdrant HNSW (`active_points_map`), atomically deleting evicted/superseded point IDs from Qdrant while persisting all revisions to Snappy-compressed **Apache Parquet** (`history_archive.parquet`) for cold `AS OF` time-travel retrieval and `117.97 ms` crash recovery.

---

## 3. Live Wikimedia Data Sources (`data/raw/` & `data/processed/`)

All experimental data was harvested from the live English Wikipedia MediaWiki API (`https://en.wikipedia.org/w/api.php?action=query&prop=revisions|extracts`) via [`src/wiki_fetcher.py`](../src/wiki_fetcher.py):
- **Raw Revision Metadata (`data/raw/wikimedia_real_revisions.json`):** **1,722 real revision records** (`revid`, `parentid`, `timestamp`, `size`, `comment`, `sha1`) across **45 tracked Wikipedia articles** spanning 5 knowledge domains:
  1. *Technology & Computing:* `Large language model`, `Apache Kafka`, `Apache Spark`, `Vector database`, `Retrieval-augmented generation`, `Python (programming language)`, `Linux kernel`, `Transformer (deep learning architecture)`, `Kubernetes`, `PostgreSQL`, `PyTorch`, `Rust (programming language)`
  2. *Science & Space:* `Artemis program`, `James Webb Space Telescope`, `SpaceX Starship`, `CRISPR gene editing`, `ITER`, `CERN`, `Messenger RNA`, `Quantum computing`, `Perseverance (rover)`
  3. *Organizations & AI Industry:* `OpenAI`, `Nvidia`, `DeepMind`, `Anthropic`, `Microsoft`, `TSMC`, `Apple Inc.`, `Meta Platforms`, `Alphabet Inc.`
  4. *Governance, Law & Global Policy:* `Artificial Intelligence Act`, `NATO`, `World Health Organization`, `European Union`, `Paris Agreement`, `United Nations Security Council`, `International Monetary Fund`, `Basel III`
  5. *Sports, Finance & Global Events:* `UEFA Champions League`, `2024 Summer Olympics`, `Formula One`, `World Chess Championship`, `Bitcoin`, `FIFA World Cup`, `S&P 500`
- **Live Article Plain-Text Extracts (`data/raw/wikimedia_real_article_extracts.json`):** Authentic Wikipedia introductory and section prose harvested from `en.wikipedia.org`.
- **Structured Multi-Chunk Benchmark (`data/processed/wikichrono_corpus.json` & `wikichrono_qa_benchmark.json`):** **60 evolving multi-chunk documents**, **246 revisions**, **1,434 chunks**, and **180 Temporal QA queries** (60 Current-State, 60 Mid-Stream Multi-Chunk Snapshot, 60 Historical Point-in-Time `AS OF`).

---

## 4. Complete Peer-Reviewed Literature Bibliography (`paper/references.bib`)

1. **Lewis et al. (NeurIPS 2020):** P. Lewis, E. Perez, A. Piktus et al., *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,"* in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 9459–9474, 2020.
2. **Malkov & Yashunin (IEEE TPAMI 2020):** Y. A. Malkov and D. A. Yashunin, *"Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs,"* *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 42, no. 4, pp. 824–836, 2020.
3. **Akidau et al. (VLDB 2015):** T. Akidau et al., *"The Dataflow Model: A Practical Approach to Balancing Correctness, Latency, and Cost in Massive-Scale, Unbounded, Out-of-Order Data Processing,"* *Proceedings of the VLDB Endowment*, vol. 8, no. 12, pp. 1792–1803, 2015.
4. **Barga et al. (CIDR 2007):** R. S. Barga, J. Goldstein, M. H. Ali, and M. Hong, *"Consistent Streaming Through Time: A Vision for Event Stream Processing,"* in *Proceedings of CIDR*, pp. 363–374, 2007.
5. **Wang et al. (SIGMOD 2021):** J. Wang et al., *"Milvus: A Purpose-Built Vector Data Management System,"* in *Proceedings of ACM SIGMOD*, pp. 2614–2627, 2021.
6. **Guo et al. (VLDB 2022):** R. Guo et al., *"Manu: A Cloud Native Vector Database Management System,"* *Proceedings of the VLDB Endowment*, vol. 15, no. 12, pp. 3548–3561, 2022.
7. **Singh et al. (ArXiv / MSR 2021):** A. Singh et al., *"FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search,"* *arXiv:2105.09613*, 2021.
8. **Xu et al. (SOSP 2023):** Y. Xu et al., *"SPFresh: Incremental In-Place Update for Billion-Scale Vector Search,"* in *Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP)*, pp. 545–561, 2023.
9. **Zhang et al. (OSDI 2023):** Q. Zhang et al., *"VBase: Unifying Online Vector Similarity Search and Relational Queries via Relaxed Monotonicity,"* in *Proceedings of USENIX OSDI*, pp. 377–395, 2023.
10. **Wu & Tan (IEEE ICDE 2015):** Y. Wu and K.-L. Tan, *"ChronoStream: Elastic Stateful Stream Computation in the Cloud,"* in *Proceedings of IEEE 31st International Conference on Data Engineering (ICDE)*, pp. 723–734, 2015. *(Disambiguated: cloud stream operator migration without vector search or LLMs).*
11. **Kim et al. (ArXiv 2025):** J. Kim et al., *"ChronoRAG: Temporal-Aware Retrieval-Augmented Generation for Long Narrative Reasoning,"* *arXiv Preprint*, 2025. *(Disambiguated: static narrative plot ordering without streaming updates or out-of-order ingestion).*
12. **Vu et al. (ACL 2024):** T. Vu et al., *"FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation,"* in *Findings of the Association for Computational Linguistics (ACL)*, 2024.
13. **Kasai et al. (NeurIPS 2023):** J. Kasai et al., *"RealTime QA: What's the Answer Right Now?,"* in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.
14. **Liška et al. (ICML 2022):** A. Liška et al., *"StreamingQA: A Benchmark for Adaptation to New Knowledge over Time in Question Answering Models,"* in *Proceedings of ICML*, pp. 13604–13622, 2022.
15. **Hou et al. (NeurIPS 2024):** Y. Hou et al., *"WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge Conflicts from Wikipedia,"* in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.
16. **Xu et al. (EMNLP 2024):** R. Xu et al., *"Knowledge Conflicts for LLMs: A Survey,"* in *Proceedings of EMNLP*, 2024.
17. **Wang et al. (ACL 2025):** F. Wang et al., *"Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models,"* in *Proceedings of ACL*, 2025.
18. **Jensen & Snodgrass (IEEE TKDE 1999):** C. S. Jensen and R. T. Snodgrass, *"Temporal Data Management,"* *IEEE Transactions on Knowledge and Data Engineering (TKDE)*, vol. 11, no. 1, pp. 36–44, 1999.
19. **Reimers & Gurevych (EMNLP 2019):** N. Reimers and I. Gurevych, *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,"* in *Proceedings of EMNLP-IJCNLP*, pp. 3982–3992, 2019.
20. **Chung et al. (JMLR 2024):** H. W. Chung et al., *"Scaling Instruction-Finetuned Language Models (Flan-T5),"* *Journal of Machine Learning Research (JMLR)*, vol. 25, no. 70, pp. 1–53, 2024.
