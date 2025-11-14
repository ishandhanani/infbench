"""
Main Streamlit Dashboard Entry Point
"""

import logging
import os

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from dashboard import components
from dashboard import (
    pareto_tab,
    latency_tab,
    node_metrics_tab,
    config_tab,
    comparison_tab,
)
from srtslurm import RunLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
components.apply_custom_css()


def render_sidebar(logs_dir, runs):
    """Render sidebar with filters and selections.
    
    Returns:
        Tuple of (filtered_runs, selected_runs, run_legend_labels, df, pareto_options, sync_info)
    """
    st.sidebar.header("Configuration")
    
    st.sidebar.divider()
    
    # Run selection with formatted labels
    st.sidebar.header("Run Selection")
    
    # Always sort by date, newest first
    sorted_runs = sorted(runs.copy(), key=lambda r: r.metadata.formatted_date, reverse=True)
    
    # Add filtering options
    st.sidebar.subheader("Filters")
    
    # 1. GPU Type Filter
    with st.sidebar.expander("🎮 GPU Type", expanded=False):
        gpu_types = set()
        for run in runs:
            gpu_type = run.metadata.gpu_type
            if gpu_type and gpu_type != "N/A":
                gpu_types.add(gpu_type)
        
        if gpu_types:
            gpu_type_options = sorted(gpu_types)
            selected_gpu_types = st.multiselect(
                "Select GPU types",
                options=gpu_type_options,
                default=gpu_type_options,
                key="gpu_type_filter",
            )
            
            if selected_gpu_types:
                sorted_runs = [r for r in sorted_runs if r.metadata.gpu_type in selected_gpu_types]
        else:
            st.caption("No GPU type information available")
    
    # 2. Topology Filter
    with st.sidebar.expander("🔧 Topology", expanded=False):
        topologies = set()
        for run in sorted_runs:
            topology = f"{run.metadata.prefill_workers}P/{run.metadata.decode_workers}D"
            topologies.add(topology)
        
        if topologies:
            topology_options = sorted(topologies)
            selected_topologies = st.multiselect(
                "Select topologies",
                options=topology_options,
                default=topology_options,
                key="topology_filter",
            )
            
            if selected_topologies:
                sorted_runs = [
                    r for r in sorted_runs
                    if f"{r.metadata.prefill_workers}P/{r.metadata.decode_workers}D" in selected_topologies
                ]
        else:
            st.caption("No topology information available")
    
    # 3. ISL/OSL Filter
    with st.sidebar.expander("📊 ISL/OSL", expanded=False):
        isl_osl_pairs = set()
        for run in sorted_runs:
            if run.profiler.isl and run.profiler.osl:
                isl_osl_pairs.add(f"{run.profiler.isl}/{run.profiler.osl}")
        
        if isl_osl_pairs:
            pair_options = sorted(isl_osl_pairs)
            selected_pairs = st.multiselect(
                "Select ISL/OSL pairs",
                options=pair_options,
                default=pair_options,
                key="isl_osl_filter",
            )
            
            if selected_pairs:
                sorted_runs = [
                    r for r in sorted_runs
                    if f"{r.profiler.isl}/{r.profiler.osl}" in selected_pairs
                ]
        else:
            st.caption("No ISL/OSL information available")
    
    # 4. Container Filter
    with st.sidebar.expander("🐳 Container", expanded=False):
        container_values = set()
        for run in sorted_runs:
            if run.metadata.container and run.metadata.container != "N/A":
                container_values.add(run.metadata.container)
        
        if container_values:
            container_options = sorted(container_values)
            selected_containers = st.multiselect(
                "Select containers",
                options=container_options,
                default=container_options,
                key="container_filter",
            )
            
            if selected_containers:
                sorted_runs = [
                    r for r in sorted_runs
                    if r.metadata.container in selected_containers or 
                       (not r.metadata.container and "N/A" in selected_containers)
                ]
        else:
            st.caption("No container information available")
    
    # Show filter results
    st.sidebar.caption(f"✅ {len(sorted_runs)} runs match filters")
    
    # Create formatted labels and mapping
    run_labels = []
    label_to_run = {}
    
    for run in sorted_runs:
        topology = f"{run.metadata.prefill_workers}P/{run.metadata.decode_workers}D"
        isl = run.profiler.isl
        osl = run.profiler.osl
        gpu_type = run.metadata.gpu_type
        gpu_suffix = f" [{gpu_type}]" if gpu_type else ""
        label = f"{topology} | {isl}/{osl}{gpu_suffix}"
        
        run_labels.append(label)
        label_to_run[label] = run
    
    # Multiselect with formatted labels
    selected_labels = st.sidebar.multiselect(
        "Select runs to compare",
        options=run_labels,
        default=run_labels[: min(3, len(run_labels))],
        help="Select one or more runs to visualize",
    )
    
    if not selected_labels:
        st.warning("Please select at least one run to visualize.")
        return None, None, None, None, None
    
    # Filter runs based on selected labels
    filtered_runs = [label_to_run[label] for label in selected_labels]
    
    # Check for incomplete runs
    incomplete_runs = [run for run in filtered_runs if not run.is_complete]
    if incomplete_runs:
        for run in incomplete_runs:
            st.warning(
                f"⚠️ **Job {run.job_id} is incomplete** - Missing concurrencies: {run.missing_concurrencies}. "
                f"Job may have failed or timed out before completing all benchmarks."
            )
    
    # Extract run IDs for graph lookups
    selected_runs = [
        f"{run.job_id}_{run.metadata.prefill_workers}P_{run.metadata.decode_workers}D_{run.metadata.run_date}"
        for run in filtered_runs
    ]
    
    # Build legend labels for graphs
    run_legend_labels = {}
    for run in filtered_runs:
        run_id = f"{run.job_id}_{run.metadata.prefill_workers}P_{run.metadata.decode_workers}D_{run.metadata.run_date}"
        topology = f"{run.metadata.prefill_workers}P/{run.metadata.decode_workers}D"
        gpu_suffix = f" [{run.metadata.gpu_type}]" if run.metadata.gpu_type else ""
        run_legend_labels[run_id] = f"Job {run.job_id} | {topology} | {run.profiler.isl}/{run.profiler.osl}{gpu_suffix}"
    
    # Get dataframe
    loader = RunLoader(logs_dir)
    df = loader.to_dataframe(filtered_runs)
    
    st.sidebar.divider()
    
    # Pareto options
    st.sidebar.header("Pareto Graph Options")
    show_cutoff = st.sidebar.checkbox("Show TPS/User cutoff line", value=False)
    cutoff_value = st.sidebar.number_input(
        "Cutoff value (TPS/User)",
        min_value=0.0,
        max_value=1000.0,
        value=30.0,
        step=1.0,
        disabled=not show_cutoff,
        help="Vertical line to mark target TPS/User threshold",
    )
    show_frontier = st.sidebar.checkbox(
        "Show Pareto Frontier",
        value=False,
        help="Highlight the efficient frontier - points where no other configuration is strictly better",
    )
    
    pareto_options = {
        "show_cutoff": show_cutoff,
        "cutoff_value": cutoff_value,
        "show_frontier": show_frontier,
    }
    
    st.sidebar.divider()
    
    # Cloud sync
    st.sidebar.header("Cloud Sync")
    auto_sync = st.sidebar.checkbox(
        "☁️ Auto-sync on load",
        value=False,
        help="Automatically pull missing runs from cloud storage on startup",
    )
    
    if st.sidebar.button("🔄 Sync Now", width="stretch"):
        components.load_data.clear()
        st.session_state["force_sync"] = True
        st.rerun()
    
    # Perform sync if enabled
    if auto_sync or st.session_state.get("force_sync", False):
        st.session_state["force_sync"] = False
        
        with st.spinner("Syncing from cloud storage..."):
            sync_performed, sync_count, error = components.sync_cloud_data(logs_dir)
        
        if sync_performed:
            if error:
                st.sidebar.error(f"Sync failed: {error}")
            elif sync_count > 0:
                st.sidebar.success(f"✓ Downloaded {sync_count} new run(s)")
            else:
                st.sidebar.info("✓ All runs up to date")
        else:
            st.sidebar.info("💡 Cloud sync not configured")
    
    return filtered_runs, selected_runs, run_legend_labels, df, pareto_options


def main():
    """Main dashboard entry point."""
    # Header
    st.markdown(
        '<div class="main-header">📊 Benchmark Analysis Dashboard</div>',
        unsafe_allow_html=True
    )
    st.markdown("Interactive visualization and analysis of benchmark logs")
    
    # Sidebar - Directory selection
    st.sidebar.header("Configuration")
    default_logs_dir = components.get_default_logs_dir()
    logs_dir = st.sidebar.text_input(
        "Logs Directory Path",
        value=default_logs_dir,
        help="Path to the directory containing benchmark log folders",
    )
    
    if not os.path.exists(logs_dir):
        st.error(f"Directory not found: {logs_dir}")
        return
    
    # Load data
    with st.spinner("Loading benchmark data..."):
        runs = components.load_data(logs_dir)
    
    if not runs:
        st.warning("No benchmark runs found in the specified directory.")
        st.info("Make sure the directory contains subdirectories with benchmark results.")
        return
    
    # Render sidebar and get filtered data
    result = render_sidebar(logs_dir, runs)
    if result[0] is None:  # No runs selected
        return
    
    filtered_runs, selected_runs, run_legend_labels, df, pareto_options = result
    
    # Summary metrics
    st.header("Summary")
    
    containers = [run.metadata.container for run in filtered_runs if run.metadata.container]
    if containers:
        unique_containers = list(set(containers))
        if len(unique_containers) == 1:
            st.caption(f"🐳 Container: {unique_containers[0]}")
        else:
            st.caption(f"🐳 Containers: {', '.join(unique_containers)}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Runs", len(selected_runs))
    with col2:
        st.metric("Data Points", len(df))
    with col3:
        st.metric("Max Throughput", f"{df['Output TPS'].max():.0f} TPS")
    with col4:
        st.metric("Max Concurrency", int(df["Concurrency"].max()))
    with col5:
        st.metric("Profilers", df["Profiler"].nunique())
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Pareto Graph",
        "⏱️ Latency Analysis",
        "🖥️ Node Metrics",
        "⚙️ Configuration",
        "🔬 Run Comparison",
    ])
    
    with tab1:
        pareto_tab.render(df, selected_runs, run_legend_labels, pareto_options)
    
    with tab2:
        latency_tab.render(df, selected_runs)
    
    with tab3:
        node_metrics_tab.render(filtered_runs, logs_dir)
    
    with tab4:
        config_tab.render(filtered_runs)
    
    with tab5:
        comparison_tab.render(filtered_runs, df, logs_dir)


if __name__ == "__main__":
    main()

