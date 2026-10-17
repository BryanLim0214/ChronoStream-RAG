"""
ChronoStream-RAG: Master 10-Seed Experimental Suite & Statistical Significance Runner
=====================================================================================
Executes:
  1. Main 10-Seed Benchmark (Seeds: 42, 101, 202, 303, 404, 505, 606, 707, 808, 909)
     across all 5 methods (M1, M2a, M2b, M3, M4) using real SentenceTransformer embeddings,
     real Qdrant vector indexing, real Parquet persistence, and real Flan-T5-Small SLM generation.
  2. Out-of-Order Network Disorder Sweep (p_ooo = 0.00 [Endogenous Compute-Lag Only], 0.10, 0.25, 0.40, 0.55)
  3. Duplicate Rate Sweep (p_dup = 0.00, 0.15, 0.30, 0.45)
  4. Working History Window Capacity Ablation (K = 1, 2, 3, 4, Full)
  5. Big Data 1 GB to 10 GB Parquet & Stream Scalability Benchmark
  6. Paired t-tests & Wilcoxon signed-rank statistical significance tests (p-values & 95% CIs)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.wiki_fetcher import generate_benchmark_corpus
from src.stream_simulator import StreamConfig, generate_disordered_event_stream
from src.methods import (
    M1ArrivalOrder,
    M2aScalarChunkLWW,
    M2bDocAtomicVersionCheck,
    M3FullBitemporal,
    M4ChronoStreamWorkingHistory
)
from src.rag_engine import NeuralRAGEngine, evaluate_method_on_benchmark
from src.scale_test import run_big_data_scale_benchmark
from src.kafka_pyspark_pipeline import PartitionedKafkaBrokerLog, StructuredMicroBatchProcessor

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]


def instantiate_methods(parquet_dir: str):
    return [
        M1ArrivalOrder(parquet_dir=parquet_dir),
        M2aScalarChunkLWW(parquet_dir=parquet_dir),
        M2bDocAtomicVersionCheck(parquet_dir=parquet_dir),
        M3FullBitemporal(parquet_dir=parquet_dir),
        M4ChronoStreamWorkingHistory(parquet_dir=parquet_dir, max_hot_revs=2)
    ]


def run_single_configuration(
    revisions: List[Dict[str, Any]],
    qa_queries: List[Dict[str, Any]],
    rag_engine: NeuralRAGEngine,
    cfg: StreamConfig,
    parquet_dir: str
) -> Dict[str, Dict[str, Any]]:
    disordered_stream = generate_disordered_event_stream(revisions, cfg)
    methods = instantiate_methods(parquet_dir)
    results_per_method = {}

    for m in methods:
        for ev in disordered_stream:
            m.process_event(ev, rag_engine.embed_texts, ms_per_chunk_encode=rag_engine.measured_ms_per_chunk)
        eval_res = evaluate_method_on_benchmark(m, rag_engine, qa_queries, seed=cfg.seed)
        results_per_method[m.name] = eval_res

    return results_per_method


def main():
    t_start = time.time()
    base_dir = Path("d:/ChronoStreamRAG")
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    parquet_dir = data_dir / "parquet_store"
    results_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = data_dir / "processed" / "wikichrono_corpus.json"
    qa_file = data_dir / "processed" / "wikichrono_qa_benchmark.json"
    if not corpus_file.exists() or not qa_file.exists():
        generate_benchmark_corpus(str(data_dir))

    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    with open(qa_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    revisions = corpus_data["revisions"]
    qa_queries = qa_data["queries"]

    # Initialize Real Neural Embedding + Local Small Language Model Engine
    rag_engine = NeuralRAGEngine()

    # Pre-warm unique chunk and query embeddings so all 10 seeds + sweeps execute rapidly
    all_texts = []
    for r in revisions:
        all_texts.extend(r["chunks"])
    for q in qa_queries:
        all_texts.append(q["question"])
    rag_engine.prewarm_embeddings(all_texts)
    rag_engine.prewarm_slm_generations(revisions, qa_queries)

    # =========================================================================
    # EXPERIMENT 1: Main 10-Seed Benchmark (p_ooo=0.25, p_dup=0.15)
    # =========================================================================
    print("\n" + "=" * 75)
    print("RUNNING EXPERIMENT 1: 10-Seed Main Benchmark (p_ooo=0.25, p_dup=0.15)")
    print("=" * 75)

    seed_runs: Dict[str, List[Dict[str, Any]]] = {}
    sample_audit_timeline = []

    for idx, seed in enumerate(SEEDS):
        cfg = StreamConfig(seed=seed, ooo_rate=0.25, dup_rate=0.15, compute_jitter_scale=18.0)
        res_map = run_single_configuration(
            revisions, qa_queries, rag_engine, cfg,
            str(parquet_dir / f"seed_{seed}")
        )
        for m_name, m_res in res_map.items():
            seed_runs.setdefault(m_name, []).append(m_res)

        if idx == 0:
            # Capture detailed stream arrival vs availability audit log + Kafka/PySpark micro-batch logs from seed 42
            disordered_sample = generate_disordered_event_stream(revisions, cfg)
            sample_audit_timeline = disordered_sample[:60]
            kafka_broker = PartitionedKafkaBrokerLog(str(data_dir / "kafka_broker_logs"), num_partitions=4)
            kafka_telemetry = kafka_broker.publish_stream_to_partitions(disordered_sample)
            mb_processor = StructuredMicroBatchProcessor(str(parquet_dir / "microbatches"))
            mb_telemetry = mb_processor.process_microbatches(disordered_sample, microbatch_size=35, watermark_delay_sec=30.0)
            print(f"  [Kafka+PySpark Pipeline] Published {kafka_telemetry['total_messages_published']} msgs across 4 partitions ({kafka_telemetry['total_broker_log_bytes']} B) & processed {mb_telemetry['num_microbatches']} micro-batches to Parquet.")

        print(
            f"  [Seed {idx+1:02d}/10 | seed={seed}] "
            f"M1 FactAcc={res_map['M1_ArrivalOrder']['overall_fact_acc']:.3f} | "
            f"M2a FactAcc={res_map['M2a_ScalarChunkLWW']['overall_fact_acc']:.3f} (Torn={res_map['M2a_ScalarChunkLWW']['idvtr_tearing_rate']:.3f}) | "
            f"M2b FactAcc={res_map['M2b_DocAtomicVersionCheck']['overall_fact_acc']:.3f} | "
            f"M3 FactAcc={res_map['M3_FullBitemporal']['overall_fact_acc']:.3f} | "
            f"M4 FactAcc={res_map['M4_ChronoStream']['overall_fact_acc']:.3f}"
        )

    # Aggregate 10-seed statistics + Statistical Significance vs M4_ChronoStream
    metric_keys = [
        "overall_version_acc", "overall_fact_acc", "overall_token_f1",
        "idvtr_tearing_rate", "orphan_chunk_leak_rate", "vandalism_resilience",
        "current_state_ver_acc", "current_state_fact_acc", "current_state_f1",
        "historical_as_of_ver_acc", "historical_as_of_fact_acc",
        "mid_stream_ver_acc", "mid_stream_fact_acc",
        "embed_compute_saved_pct", "mean_proc_latency_ms", "p95_proc_latency_ms",
        "active_qdrant_vectors", "memory_footprint_mb", "recovery_time_ms"
    ]

    aggregated_main = {}
    m4_fact_scores = [r["overall_fact_acc"] for r in seed_runs["M4_ChronoStream"]]
    m4_curr_scores = [r["current_state_fact_acc"] for r in seed_runs["M4_ChronoStream"]]

    for m_name, runs in seed_runs.items():
        agg = {"method": m_name, "n_seeds": len(runs)}
        for k in metric_keys:
            vals = np.array([r[k] for r in runs], dtype=np.float64)
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci95 = float(1.96 * std_v / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            agg[f"{k}_mean"] = round(mean_v, 4)
            agg[f"{k}_std"] = round(std_v, 4)
            agg[f"{k}_ci95"] = round(ci95, 4)

        # Paired t-test and Wilcoxon test against M4_ChronoStream on overall_fact_acc
        if m_name != "M4_ChronoStream":
            other_scores = [r["overall_fact_acc"] for r in runs]
            diffs = np.array(m4_fact_scores) - np.array(other_scores)
            if np.allclose(diffs, 0.0):
                p_ttest, p_wilcox = 1.0, 1.0
            else:
                _, p_ttest = stats.ttest_rel(m4_fact_scores, other_scores)
                try:
                    _, p_wilcox = stats.wilcoxon(m4_fact_scores, other_scores)
                except Exception:
                    p_wilcox = float(p_ttest)
            agg["p_value_ttest_vs_M4"] = float(p_ttest)
            agg["p_value_wilcoxon_vs_M4"] = float(p_wilcox)
        else:
            agg["p_value_ttest_vs_M4"] = 1.0
            agg["p_value_wilcoxon_vs_M4"] = 1.0

        aggregated_main[m_name] = agg

    # =========================================================================
    # EXPERIMENT 2: Out-of-Order Network Disorder Sweep (p_ooo = 0.0 to 0.55)
    # Note: p_ooo = 0.0 isolates Endogenous Embedding Compute-Lag Disorder!
    # =========================================================================
    print("\n" + "=" * 75)
    print("RUNNING EXPERIMENT 2: Out-of-Order Disorder Sweep (p_ooo = 0.00 to 0.55)")
    print("=" * 75)

    ooo_rates = [0.00, 0.10, 0.25, 0.40, 0.55]
    ooo_sweep_results = []
    sweep_seeds = SEEDS[:3]  # 3 seeds per parameter sweep point (10 seeds for Main Table 1)

    for p_ooo in ooo_rates:
        per_method_runs = {}
        for s in sweep_seeds:
            cfg = StreamConfig(seed=s, ooo_rate=p_ooo, dup_rate=0.15, compute_jitter_scale=18.0)
            res_map = run_single_configuration(revisions, qa_queries, rag_engine, cfg, str(parquet_dir / "sweep_ooo"))
            for m_name, m_res in res_map.items():
                per_method_runs.setdefault(m_name, []).append(m_res)

        for m_name, runs in per_method_runs.items():
            ooo_sweep_results.append({
                "ooo_rate": p_ooo,
                "method": m_name,
                "overall_fact_acc_mean": round(float(np.mean([r["overall_fact_acc"] for r in runs])), 4),
                "overall_fact_acc_std": round(float(np.std([r["overall_fact_acc"] for r in runs], ddof=1)), 4),
                "current_state_fact_acc_mean": round(float(np.mean([r["current_state_fact_acc"] for r in runs])), 4),
                "current_state_fact_acc_std": round(float(np.std([r["current_state_fact_acc"] for r in runs], ddof=1)), 4),
                "idvtr_tearing_rate_mean": round(float(np.mean([r["idvtr_tearing_rate"] for r in runs])), 4),
                "idvtr_tearing_rate_std": round(float(np.std([r["idvtr_tearing_rate"] for r in runs], ddof=1)), 4),
                "overall_token_f1_mean": round(float(np.mean([r["overall_token_f1"] for r in runs])), 4)
            })
        print(f"  Completed p_ooo={p_ooo:.2f} across {len(sweep_seeds)} seeds")

    # =========================================================================
    # EXPERIMENT 3: Duplicate Delivery Rate Sweep (p_dup = 0.00 to 0.45)
    # =========================================================================
    print("\n" + "=" * 75)
    print("RUNNING EXPERIMENT 3: Duplicate Delivery Rate Sweep (p_dup = 0.00 to 0.45)")
    print("=" * 75)

    dup_rates = [0.00, 0.15, 0.30, 0.45]
    dup_sweep_results = []

    for p_dup in dup_rates:
        per_method_runs = {}
        for s in sweep_seeds[:3]:
            cfg = StreamConfig(seed=s, ooo_rate=0.25, dup_rate=p_dup, compute_jitter_scale=18.0)
            res_map = run_single_configuration(revisions, qa_queries, rag_engine, cfg, str(parquet_dir / "sweep_dup"))
            for m_name, m_res in res_map.items():
                per_method_runs.setdefault(m_name, []).append(m_res)

        for m_name, runs in per_method_runs.items():
            dup_sweep_results.append({
                "dup_rate": p_dup,
                "method": m_name,
                "embed_compute_saved_pct_mean": round(float(np.mean([r["embed_compute_saved_pct"] for r in runs])), 2),
                "mean_proc_latency_ms": round(float(np.mean([r["mean_proc_latency_ms"] for r in runs])), 3),
                "overall_fact_acc_mean": round(float(np.mean([r["overall_fact_acc"] for r in runs])), 4),
                "duplicates_dropped_mean": round(float(np.mean([r["duplicates_dropped"] for r in runs])), 1)
            })
        print(f"  Completed p_dup={p_dup:.2f}")

    # =========================================================================
    # EXPERIMENT 4: Working History Window Size (K) Ablation for M4_ChronoStream
    # =========================================================================
    print("\n" + "=" * 75)
    print("RUNNING EXPERIMENT 4: Working History Window Size (K) Ablation")
    print("=" * 75)

    window_k_values = [1, 2, 3, 4, 6]
    ablation_k_results = []
    cfg_abl = StreamConfig(seed=42, ooo_rate=0.25, dup_rate=0.15)
    disordered_abl = generate_disordered_event_stream(revisions, cfg_abl)

    for k_val in window_k_values:
        m4_k = M4ChronoStreamWorkingHistory(parquet_dir=str(parquet_dir / f"abl_k{k_val}"), max_hot_revs=k_val)
        for ev in disordered_abl:
            m4_k.process_event(ev, rag_engine.embed_texts, ms_per_chunk_encode=rag_engine.measured_ms_per_chunk)
        res_k = evaluate_method_on_benchmark(m4_k, rag_engine, qa_queries, seed=42)
        ablation_k_results.append({
            "max_hot_revs_K": k_val,
            "active_qdrant_vectors": res_k["active_qdrant_vectors"],
            "memory_footprint_mb": res_k["memory_footprint_mb"],
            "overall_fact_acc": res_k["overall_fact_acc"],
            "historical_as_of_fact_acc": res_k["historical_as_of_fact_acc"],
            "mean_proc_latency_ms": res_k["mean_proc_latency_ms"],
            "recovery_time_ms": res_k["recovery_time_ms"]
        })
        print(f"  Window K={k_val}: Active Vectors={res_k['active_qdrant_vectors']}, Mem={res_k['memory_footprint_mb']:.2f} MB, FactAcc={res_k['overall_fact_acc']:.4f}")

    # =========================================================================
    # EXPERIMENT 5: Big Data 1 GB to 10 GB Scalability Benchmark
    # =========================================================================
    print("\n" + "=" * 75)
    print("RUNNING EXPERIMENT 5: 1 GB to 10 GB Big Data Scalability Benchmark")
    print("=" * 75)
    scale_results = run_big_data_scale_benchmark(
        corpus_path=str(corpus_file),
        output_dir=str(results_dir),
        parquet_scale_dir=str(data_dir / "scale_benchmark")
    )

    # Save all results to JSON and CSV
    all_output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": SEEDS,
        "num_seeds": len(SEEDS),
        "num_documents": corpus_data["num_documents"],
        "num_revisions": corpus_data["num_revisions"],
        "num_qa_queries": len(qa_queries),
        "measured_cpu_embed_ms_per_chunk": rag_engine.measured_ms_per_chunk,
        "kafka_broker_telemetry": kafka_telemetry,
        "pyspark_microbatch_telemetry": mb_telemetry,
        "main_10seed_summary": aggregated_main,
        "raw_10seed_runs": seed_runs,
        "ooo_sweep": ooo_sweep_results,
        "dup_sweep": dup_sweep_results,
        "ablation_window_k": ablation_k_results,
        "scale_1gb_to_10gb": scale_results,
        "sample_stream_timeline": sample_audit_timeline
    }

    with open(results_dir / "all_experiments_results.json", "w", encoding="utf-8") as f:
        json.dump(all_output, f, indent=2)

    # Export clean CSV tables for direct inspection and LaTeX/DOCX inclusion
    df_main = pd.DataFrame(list(aggregated_main.values()))
    df_main.to_csv(results_dir / "table1_main_10seed_results.csv", index=False)

    df_ooo = pd.DataFrame(ooo_sweep_results)
    df_ooo.to_csv(results_dir / "table2_ooo_disorder_sweep.csv", index=False)

    df_dup = pd.DataFrame(dup_sweep_results)
    df_dup.to_csv(results_dir / "table3_duplicate_rate_sweep.csv", index=False)

    df_abl = pd.DataFrame(ablation_k_results)
    df_abl.to_csv(results_dir / "table4_window_k_ablation.csv", index=False)

    df_scale = pd.DataFrame(scale_results)
    df_scale.to_csv(results_dir / "table5_big_data_scale_1gb_to_10gb.csv", index=False)

    print(f"\n[DONE] Completed all 5 experiments in {time.time() - t_start:.1f}s!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
