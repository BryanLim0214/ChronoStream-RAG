"""
ChronoStream-RAG: Interactive Python Dashboard (Streamlit + Standalone HTML Generator)
======================================================================================
Displays:
  - Live / Recorded RAG Answers across all 5 methods (M1, M2a, M2b, M3, M4)
  - Source Revision IDs, Event Time (t_event), Arrival Time (t_arrival), and Availability Time (t_available)
  - Intra-Document Version Tearing (IDVTR) & Orphan Chunk Leakage Diagnostics
  - 1 GB to 10 GB Big Data Scalability Telemetry (Original Content vs. Repeated Records)
  - Interactive Query Explorer & Method Comparator
"""

import json
from pathlib import Path
import pandas as pd

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def build_standalone_interactive_html(
    results_path: str = "d:/ChronoStreamRAG/results/all_experiments_results.json",
    output_html: str = "d:/ChronoStreamRAG/ChronoStream_Interactive_Dashboard.html"
):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    main = data["main_10seed_summary"]
    scale = data["scale_1gb_to_10gb"]
    timeline = data.get("sample_stream_timeline", [])[:30]
    raw_runs = data.get("raw_10seed_runs", {})

    # Extract sample query comparisons across methods from seed 42
    sample_queries = []
    if "M4_ChronoStream" in raw_runs and raw_runs["M4_ChronoStream"]:
        m4_logs = raw_runs["M4_ChronoStream"][0].get("sample_logs", [])
        m1_logs = raw_runs["M1_ArrivalOrder"][0].get("sample_logs", [])
        m2a_logs = raw_runs["M2a_ScalarChunkLWW"][0].get("sample_logs", [])
        m2b_logs = raw_runs["M2b_DocAtomicVersionCheck"][0].get("sample_logs", [])
        m3_logs = raw_runs["M3_FullBitemporal"][0].get("sample_logs", [])

        for i in range(min(len(m4_logs), 15)):
            sample_queries.append({
                "qa_id": m4_logs[i]["qa_id"],
                "query_type": m4_logs[i]["query_type"],
                "doc_id": m4_logs[i]["doc_id"],
                "expected_rev": m4_logs[i]["expected_rev"],
                "M1_revs": m1_logs[i]["retrieved_revs"],
                "M1_ans": m1_logs[i]["slm_answer"],
                "M1_ok": m1_logs[i]["fact_match"],
                "M2a_revs": m2a_logs[i]["retrieved_revs"],
                "M2a_torn": m2a_logs[i]["is_torn"],
                "M2a_ans": m2a_logs[i]["slm_answer"],
                "M2a_ok": m2a_logs[i]["fact_match"],
                "M2b_revs": m2b_logs[i]["retrieved_revs"],
                "M2b_ok": m2b_logs[i]["fact_match"],
                "M3_revs": m3_logs[i]["retrieved_revs"],
                "M3_ok": m3_logs[i]["fact_match"],
                "M4_revs": m4_logs[i]["retrieved_revs"],
                "M4_ans": m4_logs[i]["slm_answer"],
                "M4_ok": m4_logs[i]["fact_match"]
            })

    rows_html = ""
    for m_key, m_info in main.items():
        badge_color = "#27ae60" if m_key == "M4_ChronoStream" else ("#9b59b6" if m_key == "M3_FullBitemporal" else ("#3498db" if m_key == "M2b_DocAtomicVersionCheck" else "#e74c3c"))
        rows_html += f"""
        <tr>
            <td><span class="method-badge" style="background:{badge_color}">{m_key}</span></td>
            <td><strong>{m_info['overall_fact_acc_mean']*100:.2f}%</strong> ± {m_info['overall_fact_acc_std']*100:.2f}%</td>
            <td>{m_info['current_state_fact_acc_mean']*100:.2f}%</td>
            <td>{m_info['historical_as_of_fact_acc_mean']*100:.2f}%</td>
            <td>{m_info['idvtr_tearing_rate_mean']*100:.2f}%</td>
            <td>{m_info['vandalism_resilience_mean']*100:.2f}%</td>
            <td>{m_info['embed_compute_saved_pct_mean']:.1f}%</td>
            <td>{m_info['mean_proc_latency_ms_mean']:.2f} ms</td>
            <td>{int(m_info['active_qdrant_vectors_mean'])}</td>
            <td>{m_info['recovery_time_ms_mean']:.1f} ms</td>
        </tr>
        """

    qa_rows_html = ""
    for sq in sample_queries:
        m1_badge = "✅" if sq["M1_ok"] else "❌ Stale/Inverted"
        m2a_badge = "✅" if sq["M2a_ok"] else ("⚠️ TORN CHUNKS" if sq["M2a_torn"] else "❌ Stale")
        m2b_badge = "✅" if sq["M2b_ok"] else "❌ Dropped History"
        m4_badge = "✅ Atomic Match" if sq["M4_ok"] else "❌"
        qa_rows_html += f"""
        <tr>
            <td><code>{sq['qa_id']}</code></td>
            <td><span class="qtype-pill">{sq['query_type']}</span></td>
            <td><code>{sq['doc_id']}</code></td>
            <td><strong>v{sq['expected_rev']}</strong></td>
            <td>v{sq['M1_revs']} ({m1_badge})</td>
            <td>v{sq['M2a_revs']} ({m2a_badge})</td>
            <td>v{sq['M2b_revs']} ({m2b_badge})</td>
            <td>v{sq['M4_revs']} ({m4_badge})</td>
            <td style="font-size:0.84rem;color:#cbd5e1;">{sq['M4_ans']}</td>
        </tr>
        """

    timeline_rows_html = ""
    t0 = min((e["t_event"] for e in timeline), default=0.0)
    for ev in timeline[:15]:
        status_tag = "🔁 DUPLICATE" if ev["is_duplicate_delivery"] else ("⏳ DELAYED OOO" if ev["is_network_delayed"] else "⚡ IN-ORDER")
        status_cls = "tag-dup" if ev["is_duplicate_delivery"] else ("tag-ooo" if ev["is_network_delayed"] else "tag-ok")
        timeline_rows_html += f"""
        <tr>
            <td><code>{ev['stream_msg_id']}</code></td>
            <td>{ev['title']}</td>
            <td><strong>v{ev['rev_num']}</strong> (<code>{ev['wiki_revid']}</code>)</td>
            <td>+{ev['t_event'] - t0:.2f}s</td>
            <td>+{ev['t_arrival'] - t0:.2f}s (Δ={ev['network_delay_sec']:.1f}s)</td>
            <td>+{ev['t_embed_complete'] - t0:.2f}s</td>
            <td>{ev['num_chunks']} chunks</td>
            <td><span class="status-pill {status_cls}">{status_tag}</span></td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ChronoStream-RAG Interactive Telemetry & Evaluation Dashboard</title>
<style>
  :root {{
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #10b981;
    --blue: #3b82f6;
    --purple: #8b5cf6;
    --red: #ef4444;
    --orange: #f59e0b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 28px 36px;
    line-height: 1.5;
  }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 18px;
    margin-bottom: 24px;
  }}
  h1 {{ font-size: 1.75rem; font-weight: 700; color: #fff; }}
  .subtitle {{ color: var(--muted); font-size: 0.95rem; margin-top: 4px; }}
  .authors-box {{
    text-align: right;
    font-size: 0.85rem;
    color: var(--muted);
    background: var(--card);
    padding: 10px 16px;
    border-radius: 8px;
    border: 1px solid var(--border);
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 26px;
  }}
  .kpi-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px;
  }}
  .kpi-title {{ font-size: 0.82rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi-val {{ font-size: 1.85rem; font-weight: 800; margin: 6px 0; color: var(--accent); }}
  .kpi-sub {{ font-size: 0.8rem; color: #cbd5e1; }}
  .section-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 26px;
  }}
  .section-title {{
    font-size: 1.18rem;
    font-weight: 700;
    margin-bottom: 14px;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  th, td {{
    padding: 11px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    background: #0f172a;
    color: var(--muted);
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
  }}
  tr:hover td {{ background: rgba(255,255,255,0.03); }}
  .method-badge {{
    padding: 4px 9px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.8rem;
    color: #fff;
    display: inline-block;
  }}
  .qtype-pill {{
    background: #334155;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.78rem;
    color: #38bdf8;
  }}
  .status-pill {{
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
  }}
  .tag-ok {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
  .tag-ooo {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}
  .tag-dup {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; }}
  .gallery-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 18px;
  }}
  .gallery-item {{
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
  }}
  .gallery-item img {{
    width: 100%;
    border-radius: 6px;
    background: #fff;
  }}
  .gallery-caption {{
    margin-top: 8px;
    font-size: 0.85rem;
    color: var(--muted);
    font-weight: 600;
  }}
</style>
</head>
<body>
  <header>
    <div>
      <h1>⏱️ ChronoStream-RAG: Live Bitemporal & Document-Atomic Streaming RAG Dashboard</h1>
      <div class="subtitle">Testing & Preventing Out-of-Order Inversions, Cross-Chunk Version Tearing, and Vandalism Drift in Streaming Vector Search</div>
    </div>
    <div class="authors-box">
      <strong>Big Data Analytics Group Project</strong><br>
      Lynette Osako (8000290404) • Panhapiseth Lim (8000265274) • Rajan Yadav (8000277867)
    </div>
  </header>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-title">ChronoStream Fact Agreement</div>
      <div class="kpi-val">{main['M4_ChronoStream']['overall_fact_acc_mean']*100:.2f}%</div>
      <div class="kpi-sub">vs. {main['M1_ArrivalOrder']['overall_fact_acc_mean']*100:.1f}% (M1) & {main['M2a_ScalarChunkLWW']['overall_fact_acc_mean']*100:.1f}% (M2a)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Cross-Chunk Version Tearing</div>
      <div class="kpi-val" style="color:#38bdf8;">0.00%</div>
      <div class="kpi-sub">Eliminates {main['M2a_ScalarChunkLWW']['idvtr_tearing_rate_mean']*100:.1f}% Scalar LWW Tearing</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Embedding Compute Saved</div>
      <div class="kpi-val" style="color:#a855f7;">{main['M4_ChronoStream']['embed_compute_saved_pct_mean']:.1f}%</div>
      <div class="kpi-sub">Via Chunk-Level SHA-256 Delta Dedup</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">HNSW Index Vector Reduction</div>
      <div class="kpi-val" style="color:#f59e0b;">-{100.0*(main['M3_FullBitemporal']['active_qdrant_vectors_mean']-main['M4_ChronoStream']['active_qdrant_vectors_mean'])/main['M3_FullBitemporal']['active_qdrant_vectors_mean']:.1f}%</div>
      <div class="kpi-sub">{int(main['M4_ChronoStream']['active_qdrant_vectors_mean'])} hot vecs vs. {int(main['M3_FullBitemporal']['active_qdrant_vectors_mean'])} in Full Bitemporal</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Post-Crash Recovery Latency</div>
      <div class="kpi-val" style="color:#10b981;">{main['M4_ChronoStream']['recovery_time_ms_mean']:.1f} ms</div>
      <div class="kpi-sub">Parquet-WAL + Watermark Checkpoint</div>
    </div>
  </div>

  <div class="section-card">
    <div class="section-title">📊 10-Seed Experimental Comparison Across All Streaming RAG Consistency Methods (N=1,800 Queries/Method)</div>
    <table>
      <thead>
        <tr>
          <th>Method</th>
          <th>Overall Fact Agreement</th>
          <th>Current-State Acc</th>
          <th>Historical AS-OF Acc</th>
          <th>Version Tearing (IDVTR)</th>
          <th>Vandalism Resilience</th>
          <th>Embed Compute Saved</th>
          <th>Mean Update Latency</th>
          <th>Active Qdrant Vectors</th>
          <th>Crash Recovery Time</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <div class="section-card">
    <div class="section-title">🔍 Live RAG Answer, Source Version & Cross-Chunk Tearing Inspector (Seed 42 Audit)</div>
    <table>
      <thead>
        <tr>
          <th>Query ID</th>
          <th>Query Category</th>
          <th>Document ID</th>
          <th>Truth Rev</th>
          <th>M1 (Arrival Order)</th>
          <th>M2a (Scalar Chunk LWW)</th>
          <th>M2b (Doc-Atomic Head)</th>
          <th>M4 (ChronoStream-RAG)</th>
          <th>ChronoStream-RAG Generated SLM Output</th>
        </tr>
      </thead>
      <tbody>
        {qa_rows_html}
      </tbody>
    </table>
  </div>

  <div class="section-card">
    <div class="section-title">⏱️ Bitemporal Event-Stream Audit Log: Source Event Time ($t_{{event}}$) vs. Broker Arrival ($t_{{arrival}}$) vs. Index Availability ($t_{{available}}$)</div>
    <table>
      <thead>
        <tr>
          <th>Stream Message ID</th>
          <th>Wikimedia Document</th>
          <th>Source Version (RevID)</th>
          <th>Event Time ($t_{{event}}$)</th>
          <th>Broker Arrival ($t_{{arrival}}$)</th>
          <th>Qdrant Available ($t_{{available}}$)</th>
          <th>Structure</th>
          <th>Stream Condition</th>
        </tr>
      </thead>
      <tbody>
        {timeline_rows_html}
      </tbody>
    </table>
  </div>

  <div class="section-card">
    <div class="section-title">📈 Visual Analytics & Multi-Perspective Publication Charts Gallery</div>
    <div class="gallery-grid">
      <div class="gallery-item">
        <img src="plots/fig01_main_accuracy_comparison.png" alt="Main Accuracy">
        <div class="gallery-caption">Figure 1: Overall Version Selection Accuracy, Fact Agreement, and SLM Token F1 (10 Seeds)</div>
      </div>
      <div class="gallery-item">
        <img src="plots/fig03_anomaly_taxonomy_tearing_vandalism.png" alt="Anomaly Taxonomy">
        <div class="gallery-caption">Figure 3: Intra-Document Version Tearing (IDVTR), Orphan Chunk Leakage & Vandalism Immunity</div>
      </div>
      <div class="gallery-item">
        <img src="plots/fig04_ooo_disorder_sweep.png" alt="Out of Order Sweep">
        <div class="gallery-caption">Figure 4: Resilience Across Network Disorder Rates & Endogenous Compute-Lag Inversion at p_ooo=0%</div>
      </div>
      <div class="gallery-item">
        <img src="plots/fig09_big_data_scale_1gb_to_10gb.png" alt="Big Data Scale">
        <div class="gallery-caption">Figure 9: 1 GB to 10 GB Big Data Scalability (Separating Original Content from Repeated Records)</div>
      </div>
      <div class="gallery-item">
        <img src="plots/fig13_bitemporal_error_attribution.png" alt="Bitemporal Error Attribution">
        <div class="gallery-caption">Figure 13: Perspective 1 — Live Interleaved Bitemporal Error Attribution (Unarrived Lag vs. Held-Data Index Error)</div>
      </div>
      <div class="gallery-item">
        <img src="plots/fig14_cross_model_flan_vs_distilbert.png" alt="Cross Model Flan vs DistilBERT">
        <div class="gallery-caption">Figure 14: Perspective 2 — Cross-Architecture Neural Reader Comparison (Generative Flan-T5-Small vs. Extractive DistilBERT-SQuAD)</div>
      </div>
      <div class="gallery-item">
        <img src="plots/fig15_edit_taxonomy_breakdown.png" alt="Edit Taxonomy Breakdown">
        <div class="gallery-caption">Figure 15: Perspective 3 — Performance Breakdown Across Wikimedia Edit Taxonomy (6->6 vs. 6->4 Consolidation, Vandalism, AS-OF)</div>
      </div>
      <div class="gallery-item">
        <img src="plots/fig11_domain_heatmap.png" alt="Domain Heatmap">
        <div class="gallery-caption">Figure 11: Perspective 4 — Empirical Fact Agreement Accuracy Across 5 Wikimedia Knowledge Domains</div>
      </div>
    </div>
  </div>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[Dashboard] Built standalone interactive HTML dashboard at {output_html}")


def run_streamlit_app():
    st.set_page_config(page_title="ChronoStream-RAG Dashboard", layout="wide")
    st.title("⏱️ ChronoStream-RAG: Bitemporal Streaming RAG Dashboard")
    st.caption("Authors: Lynette Osako (8000290404) • Panhapiseth Lim (8000265274) • Rajan Yadav (8000277867)")

    res_path = Path("d:/ChronoStreamRAG/results/all_experiments_results.json")
    if not res_path.exists():
        st.warning("Run `python experiments/run_all_experiments.py` first to populate telemetry.")
        return

    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    main = data["main_10seed_summary"]
    cols = st.columns(4)
    cols[0].metric("ChronoStream Fact Accuracy", f"{main['M4_ChronoStream']['overall_fact_acc_mean']*100:.2f}%")
    cols[1].metric("Cross-Chunk Tearing (IDVTR)", "0.00%", delta=f"-{main['M2a_ScalarChunkLWW']['idvtr_tearing_rate_mean']*100:.1f}% vs Scalar LWW")
    cols[2].metric("Embedding Compute Saved", f"{main['M4_ChronoStream']['embed_compute_saved_pct_mean']:.1f}%")
    cols[3].metric("Post-Crash Recovery Time", f"{main['M4_ChronoStream']['recovery_time_ms_mean']:.1f} ms")

    st.subheader("10-Seed Method Comparison Table")
    st.dataframe(pd.DataFrame(list(main.values())))

    st.subheader("1 GB to 10 GB Scalability (Original Content vs. Repeated Records)")
    st.dataframe(pd.DataFrame(data["scale_1gb_to_10gb"]))


if __name__ == "__main__":
    build_standalone_interactive_html()
