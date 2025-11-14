"""
Run Comparison Tab (Isolation Mode)
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from srtslurm.config_reader import get_all_configs
from srtslurm.run_comparison import (
    calculate_summary_scorecard,
    compare_configs,
    compare_metrics,
    get_delta_data_for_graphs,
)
from srtslurm.visualizations import create_pareto_graph, create_latency_vs_concurrency_graph


def render(filtered_runs: list, df: pd.DataFrame, logs_dir: str):
    """Render the run comparison tab.
    
    Args:
        filtered_runs: List of BenchmarkRun objects
        df: DataFrame with benchmark data
        logs_dir: Path to logs directory
    """
    st.subheader("🔬 Run Comparison (Isolation Mode)")
    st.markdown("""
    Compare two benchmark runs in detail to understand configuration changes and performance deltas.
    Select exactly 2 runs to begin comparison.
    """)
    
    # Run Selection Panel
    st.markdown("### 📋 Select Runs to Compare")
    
    # Add sort control
    sort_col1, sort_col2 = st.columns([1, 3])
    with sort_col1:
        sort_by = st.selectbox(
            "Sort by",
            options=["Newest First", "Oldest First", "Job ID"],
            key="comparison_sort"
        )
    
    # Sort filtered_runs
    comparison_sorted = filtered_runs.copy()
    if sort_by == "Newest First":
        comparison_sorted = sorted(comparison_sorted, key=lambda r: r.metadata.formatted_date, reverse=True)
    elif sort_by == "Oldest First":
        comparison_sorted = sorted(comparison_sorted, key=lambda r: r.metadata.formatted_date)
    elif sort_by == "Job ID":
        comparison_sorted = sorted(comparison_sorted, key=lambda r: r.job_id)
    
    # Create formatted labels
    run_options = []
    run_map = {}
    
    for run in comparison_sorted:
        date_short = run.metadata.formatted_date
        topology = f"{run.metadata.prefill_workers}P/{run.metadata.decode_workers}D"
        label = f"[{date_short}] Job {run.job_id} - {topology} (ISL {run.profiler.isl})"
        run_options.append(label)
        run_map[label] = run
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Run A (Baseline)**")
        run_a_label = st.selectbox("Select baseline run", options=run_options, key="run_a_select")
        run_a = run_map.get(run_a_label)
        
        if run_a:
            st.caption(f"📅 {run_a.metadata.formatted_date}")
            st.caption(f"📊 ISL/OSL: {run_a.profiler.isl}/{run_a.profiler.osl}")
            st.caption(f"🖥️ Topology: {run_a.metadata.prefill_workers}P/{run_a.metadata.decode_workers}D")
            st.caption(f"🎯 Total GPUs: {run_a.total_gpus}")
            
            if not run_a.is_complete:
                st.warning(f"⚠️ **Incomplete job** - Missing concurrencies: {run_a.missing_concurrencies}")
    
    with col2:
        st.markdown("**Run B (Comparison)**")
        run_b_label = st.selectbox("Select comparison run", options=run_options, key="run_b_select")
        run_b = run_map.get(run_b_label)
        
        if run_b:
            st.caption(f"📅 {run_b.metadata.formatted_date}")
            st.caption(f"📊 ISL/OSL: {run_b.profiler.isl}/{run_b.profiler.osl}")
            st.caption(f"🖥️ Topology: {run_b.metadata.prefill_workers}P/{run_b.metadata.decode_workers}D")
            st.caption(f"🎯 Total GPUs: {run_b.total_gpus}")
            
            if not run_b.is_complete:
                st.warning(f"⚠️ **Incomplete job** - Missing concurrencies: {run_b.missing_concurrencies}")
    
    # Validation
    if run_a_label == run_b_label:
        st.warning("⚠️ Please select two different runs to compare.")
    elif run_a and run_b:
        _render_comparison(run_a, run_b, df)


def _render_comparison(run_a, run_b, df):
    """Render the detailed comparison between two runs."""
    st.divider()
    
    # Get config data for both runs
    run_a_path = run_a.metadata.path
    run_b_path = run_b.metadata.path
    
    configs_a = get_all_configs(run_a_path)
    configs_b = get_all_configs(run_b_path)
    
    if not configs_a or not configs_b:
        st.error("Could not load configuration files for one or both runs.")
        return
    
    config_a = configs_a[0]
    config_b = configs_b[0]
    
    # Perform comparison
    config_comparison = compare_configs(config_a, config_b)
    metrics_comparison = compare_metrics(run_a, run_b)
    scorecard = calculate_summary_scorecard(metrics_comparison)
    
    # Display Configuration Comparison
    with st.expander("📋 Configuration Differences", expanded=True):
        st.markdown("#### Topology Summary")
        
        topo_df = pd.DataFrame({
            "Parameter": ["Prefill TP", "Prefill DP", "Model", "Context Length", "Container"],
            "Run A": [
                str(config_comparison["topology_summary"]["prefill_tp"][0]),
                str(config_comparison["topology_summary"]["prefill_dp"][0]),
                str(config_comparison["topology_summary"]["model"][0]),
                str(config_comparison["topology_summary"]["context_length"][0]),
                str(run_a.metadata.container or "N/A"),
            ],
            "Run B": [
                str(config_comparison["topology_summary"]["prefill_tp"][1]),
                str(config_comparison["topology_summary"]["prefill_dp"][1]),
                str(config_comparison["topology_summary"]["model"][1]),
                str(config_comparison["topology_summary"]["context_length"][1]),
                str(run_b.metadata.container or "N/A"),
            ],
        })
        st.dataframe(topo_df, width="stretch", hide_index=True)
        
        st.divider()
        
        num_diffs = config_comparison["num_differences"]
        if num_diffs > 0:
            st.markdown(f"#### ⚠️ **{num_diffs}** Configuration Flags Differ")
            
            # Group by category
            diff_df = pd.DataFrame(config_comparison["flag_differences"])
            
            if not diff_df.empty:
                for category in sorted(diff_df["category"].unique()):
                    category_diffs = diff_df[diff_df["category"] == category]
                    st.markdown(f"**{category}**")
                    
                    display_df = category_diffs[["flag", "run_a_value", "run_b_value"]]
                    display_df.columns = ["Flag", "Run A", "Run B"]
                    
                    st.dataframe(display_df, width="stretch", hide_index=True)
        else:
            st.success("✅ No configuration differences detected!")
        
        # Identical flags (collapsed)
        num_identical = len(config_comparison["identical_flags"])
        if num_identical > 0:
            with st.expander(f"Show {num_identical} Identical Flags"):
                identical_df = pd.DataFrame(config_comparison["identical_flags"])
                identical_df["value"] = identical_df["value"].astype(str)
                st.dataframe(identical_df, width="stretch", hide_index=True)
    
    st.divider()
    
    # Display Performance Comparison
    st.markdown("### 📊 Performance Comparison")
    
    if not metrics_comparison.empty:
        # Summary scorecard
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Metrics Improved", scorecard["num_improved"])
        with col2:
            st.metric("⚠️ Metrics Regressed", scorecard["num_regressed"])
        with col3:
            st.metric("➡️ Unchanged", scorecard["num_unchanged"])
        
        if scorecard["biggest_improvement"]:
            st.success(f"**Biggest Win:** {scorecard['biggest_improvement']}")
        if scorecard["biggest_regression"]:
            st.warning(f"**Biggest Loss:** {scorecard['biggest_regression']}")
        
        st.divider()
        
        # Detailed metrics table
        st.markdown("#### Detailed Metrics Comparison")
        
        display_df = metrics_comparison.copy()
        display_df["Status"] = display_df["Improved"].apply(lambda x: "✅" if x else "⚠️")
        display_df = display_df[["Concurrency", "Metric", "Run A", "Run B", "Delta", "% Change", "Status"]]
        
        st.dataframe(
            display_df.style.format({
                "Run A": "{:.2f}",
                "Run B": "{:.2f}",
                "Delta": "{:.2f}",
                "% Change": "{:.1f}%",
            }).map(
                lambda x: "background-color: #d4edda" if x == "✅"
                else ("background-color: #f8d7da" if x == "⚠️" else ""),
                subset=["Status"],
            ),
            width="stretch",
            height=400,
        )
        
        # Visual Comparison Graphs
        st.divider()
        st.markdown("### 📈 Visual Comparison")
        
        viz_tab1, viz_tab2 = st.tabs(["Side-by-Side Comparison", "Delta Graphs"])
        
        with viz_tab1:
            _render_overlay_comparison(run_a, run_b, df)
        
        with viz_tab2:
            _render_delta_graphs(run_a, run_b)
    else:
        st.warning("No matching concurrency levels found between the two runs to compare.")


def _render_overlay_comparison(run_a, run_b, df):
    """Render overlay comparison graphs."""
    st.markdown("#### Overlay Comparison")
    
    run_a_id = f"{run_a.job_id}_{run_a.metadata.prefill_workers}P_{run_a.metadata.decode_workers}D_{run_a.metadata.run_date}"
    run_b_id = f"{run_b.job_id}_{run_b.metadata.prefill_workers}P_{run_b.metadata.decode_workers}D_{run_b.metadata.run_date}"
    selected_for_comparison = [run_a_id, run_b_id]
    comparison_df = df[df["Run ID"].isin(selected_for_comparison)]
    
    # Build legend labels
    comparison_labels = {}
    for r in [run_a, run_b]:
        r_id = f"{r.job_id}_{r.metadata.prefill_workers}P_{r.metadata.decode_workers}D_{r.metadata.run_date}"
        topology = f"{r.metadata.prefill_workers}P/{r.metadata.decode_workers}D"
        gpu_suffix = f" [{r.metadata.gpu_type}]" if r.metadata.gpu_type else ""
        comparison_labels[r_id] = f"Job {r.job_id} | {topology} | {r.profiler.isl}/{r.profiler.osl}{gpu_suffix}"
    
    if not comparison_df.empty:
        # Pareto comparison
        st.markdown("**Pareto Frontier**")
        pareto_comparison_fig = create_pareto_graph(
            comparison_df,
            selected_for_comparison,
            show_cutoff=False,
            cutoff_value=30.0,
            show_frontier=False,
            run_labels=comparison_labels,
        )
        st.plotly_chart(pareto_comparison_fig, width="stretch", key="comparison_pareto")
        
        # Latency comparisons
        st.markdown("**Latency Metrics**")
        latency_metrics = [
            ("Time to First Token (TTFT)", "Mean TTFT (ms)", "TTFT (ms)"),
            ("Time Per Output Token (TPOT)", "Mean TPOT (ms)", "TPOT (ms)"),
            ("Inter-Token Latency (ITL)", "Mean ITL (ms)", "ITL (ms)"),
        ]
        
        for metric_name, metric_col, y_label in latency_metrics:
            fig = create_latency_vs_concurrency_graph(
                comparison_df,
                selected_for_comparison,
                metric_name,
                metric_col,
                y_label,
            )
            st.plotly_chart(fig, width="stretch", key=f"comparison_latency_{metric_name}")


def _render_delta_graphs(run_a, run_b):
    """Render delta analysis graphs."""
    st.markdown("#### Delta Analysis (Run B - Run A)")
    st.caption("Positive values indicate Run B is higher. For latency, negative is better. For throughput, positive is better.")
    
    delta_data = get_delta_data_for_graphs(run_a, run_b)
    
    if not delta_data.empty:
        # TTFT Delta
        if "TTFT Delta (ms)" in delta_data.columns:
            fig_ttft = go.Figure()
            fig_ttft.add_trace(
                go.Scatter(
                    x=delta_data["Concurrency"],
                    y=delta_data["TTFT Delta (ms)"],
                    mode="lines+markers",
                    name="TTFT Delta",
                    line={"color": "blue", "width": 2},
                    marker={"size": 8},
                )
            )
            fig_ttft.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No change")
            fig_ttft.update_layout(
                title="TTFT Delta (Run B - Run A)",
                xaxis_title="Concurrency",
                yaxis_title="Delta TTFT (ms)",
                height=400,
            )
            st.plotly_chart(fig_ttft, width="stretch", key="delta_ttft")
        
        # TPOT Delta
        if "TPOT Delta (ms)" in delta_data.columns:
            fig_tpot = go.Figure()
            fig_tpot.add_trace(
                go.Scatter(
                    x=delta_data["Concurrency"],
                    y=delta_data["TPOT Delta (ms)"],
                    mode="lines+markers",
                    name="TPOT Delta",
                    line={"color": "green", "width": 2},
                    marker={"size": 8},
                )
            )
            fig_tpot.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No change")
            fig_tpot.update_layout(
                title="TPOT Delta (Run B - Run A)",
                xaxis_title="Concurrency",
                yaxis_title="Delta TPOT (ms)",
                height=400,
            )
            st.plotly_chart(fig_tpot, width="stretch", key="delta_tpot")
        
        # Throughput Delta
        if "Throughput Delta (TPS)" in delta_data.columns:
            fig_tps = go.Figure()
            fig_tps.add_trace(
                go.Scatter(
                    x=delta_data["Concurrency"],
                    y=delta_data["Throughput Delta (TPS)"],
                    mode="lines+markers",
                    name="Throughput Delta",
                    line={"color": "red", "width": 2},
                    marker={"size": 8},
                )
            )
            fig_tps.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No change")
            fig_tps.update_layout(
                title="Output Throughput Delta (Run B - Run A)",
                xaxis_title="Concurrency",
                yaxis_title="Delta Throughput (tokens/s)",
                height=400,
            )
            st.plotly_chart(fig_tps, width="stretch", key="delta_throughput")
    else:
        st.info("No matching concurrency levels between the two runs.")
