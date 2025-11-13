"""
Streamlit Dashboard for Benchmark Log Analysis
"""

import logging
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from srtslurm import (
    NodeAnalyzer,
    RunLoader,
    format_config_for_display,
    parse_command_line_from_err,
)
from srtslurm.config_reader import (
    get_all_configs,
    get_command_line_args,
    get_environment_variables,
    parse_command_line_to_dict,
)
from srtslurm.run_comparison import (
    calculate_summary_scorecard,
    compare_configs,
    compare_metrics,
    get_delta_data_for_graphs,
)
from srtslurm.visualizations import (
    calculate_pareto_frontier,
    create_latency_vs_concurrency_graph,
    create_node_metric_graph,
    create_pareto_graph,
    create_stacked_metric_graph,
)

# Configure logging to show validation warnings
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

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(logs_dir):
    """Load and cache benchmark data."""
    loader = RunLoader(logs_dir)
    runs = loader.load_all()

    # Convert to dicts for compatibility with existing code
    return [_run_to_dict(run) for run in runs]


def _run_to_dict(run) -> dict:
    """Convert BenchmarkRun object to dict format.

    Temporary converter to maintain compatibility with existing app.py code.
    Can be removed once app.py is fully migrated to use objects directly.
    """
    from datetime import datetime

    # Format date to match legacy format
    try:
        dt = datetime.strptime(run.metadata.run_date, "%Y%m%d_%H%M%S")
        formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        formatted_date = run.metadata.run_date

    return {
        "slurm_job_id": f"{run.job_id}_{run.metadata.prefill_nodes}P_{run.metadata.decode_nodes}D_{run.metadata.run_date}",
        "path": run.metadata.path,
        "run_date": formatted_date,
        "container": run.metadata.container,
        "prefill_dp": run.metadata.prefill_nodes,
        "decode_dp": run.metadata.decode_nodes,
        "prefill_tp": run.metadata.gpus_per_node,
        "decode_tp": run.metadata.gpus_per_node,
        "frontends": run.metadata.num_additional_frontends,
        "profiler_type": run.profiler.profiler_type,
        "isl": run.profiler.isl,
        "osl": run.profiler.osl,
        "concurrencies": run.profiler.concurrency_values,
        "output_tps": run.profiler.output_tps,
        "total_tps": run.profiler.total_tps,
        "mean_itl_ms": run.profiler.mean_itl_ms,
        "mean_ttft_ms": run.profiler.mean_ttft_ms,
        "mean_tpot_ms": run.profiler.mean_tpot_ms,
        "request_rate": run.profiler.request_rate,
    }


@st.cache_data(show_spinner="Loading node metrics from logs...")
def load_node_metrics(run_path: str):
    """Load and cache node metrics from .err files.

    Args:
        run_path: Path to the run directory

    Returns:
        List of parsed node metrics (as dicts for compatibility)
    """
    analyzer = NodeAnalyzer()
    nodes = analyzer.parse_run_logs(run_path)

    # Convert to dicts for compatibility with existing visualization code
    return [_node_to_dict(node) for node in nodes]


def _node_to_dict(node) -> dict:
    """Convert NodeMetrics object to dict format.

    Temporary converter for compatibility with existing visualization code.
    """
    return {
        "node_info": node.node_info,
        "prefill_batches": [_batch_to_dict(b) for b in node.batches],
        "memory_snapshots": [_memory_to_dict(m) for m in node.memory_snapshots],
        "config": node.config,
        "run_id": node.run_id,
    }


def _batch_to_dict(batch) -> dict:
    """Convert BatchMetrics to dict."""
    result = {
        "timestamp": batch.timestamp,
        "dp": batch.dp,
        "tp": batch.tp,
        "ep": batch.ep,
        "type": batch.batch_type,
    }
    # Add optional fields
    for field in [
        "new_seq",
        "new_token",
        "cached_token",
        "token_usage",
        "running_req",
        "queue_req",
        "prealloc_req",
        "inflight_req",
        "input_throughput",
        "gen_throughput",
        "transfer_req",
        "num_tokens",
        "preallocated_usage",
    ]:
        value = getattr(batch, field)
        if value is not None:
            result[field] = value
    return result


def _memory_to_dict(mem) -> dict:
    """Convert MemoryMetrics to dict."""
    result = {
        "timestamp": mem.timestamp,
        "dp": mem.dp,
        "tp": mem.tp,
        "ep": mem.ep,
        "type": mem.metric_type,
    }
    # Add optional fields
    for field in ["avail_mem_gb", "mem_usage_gb", "kv_cache_gb", "kv_tokens"]:
        value = getattr(mem, field)
        if value is not None:
            result[field] = value
    return result


def _runs_to_dataframe(run_dicts: list[dict]):
    """Convert list of run dicts to DataFrame.

    Temporary wrapper that uses old metrics.py logic.
    TODO: Could be removed by converting run_dicts back to BenchmarkRun objects.
    """
    import pandas as pd

    rows = []

    for run in run_dicts:
        # Calculate total GPUs
        total_gpus = 0
        if "prefill_tp" in run and "prefill_dp" in run:
            total_gpus += run["prefill_tp"] * run["prefill_dp"]
        if "decode_tp" in run and "decode_dp" in run:
            total_gpus += run["decode_tp"] * run["decode_dp"]
        total_gpus = total_gpus if total_gpus > 0 else 1

        run_id = run.get("slurm_job_id", "Unknown")
        output_tps = run.get("output_tps", [])
        total_tps = run.get("total_tps", [])
        concurrencies = run.get("concurrencies", [])

        # Create a row for each concurrency level
        for i in range(len(output_tps)):
            tps = output_tps[i]
            output_tps_per_gpu = tps / total_gpus
            
            # Get total TPS for this concurrency level
            total_token_tps = total_tps[i] if i < len(total_tps) else None
            total_tps_per_gpu = total_token_tps / total_gpus if total_token_tps else None

            tpot = run.get("mean_tpot_ms", [])[i] if i < len(run.get("mean_tpot_ms", [])) else None
            tps_per_user = 1000 / tpot if tpot and tpot > 0 else 0

            row = {
                "Run ID": run_id,
                "Run Date": run.get("run_date", "N/A"),
                "Profiler": run.get("profiler_type", "N/A"),
                "ISL": run.get("isl", "N/A"),
                "OSL": run.get("osl", "N/A"),
                "Prefill TP": run.get("prefill_tp", "N/A"),
                "Prefill DP": run.get("prefill_dp", "N/A"),
                "Decode TP": run.get("decode_tp", "N/A"),
                "Decode DP": run.get("decode_dp", "N/A"),
                "Frontends": run.get("frontends", "N/A"),
                "Total GPUs": total_gpus,
                "Request Rate": run.get("request_rate", [])[i]
                if i < len(run.get("request_rate", []))
                else "N/A",
                "Concurrency": concurrencies[i] if i < len(concurrencies) else "N/A",
                "Output TPS": tps,
                "Total TPS": total_token_tps if total_token_tps else "N/A",
                "Output TPS/GPU": output_tps_per_gpu,
                "Total TPS/GPU": total_tps_per_gpu if total_tps_per_gpu else "N/A",
                "Output TPS/User": tps_per_user,
                "Mean TTFT (ms)": run.get("mean_ttft_ms", [])[i]
                if i < len(run.get("mean_ttft_ms", []))
                else "N/A",
                "Mean TPOT (ms)": tpot if tpot else "N/A",
                "Mean ITL (ms)": run.get("mean_itl_ms", [])[i]
                if i < len(run.get("mean_itl_ms", []))
                else "N/A",
            }
            rows.append(row)

    return pd.DataFrame(rows)


# Wrapper functions to use generic graph builders with caching
@st.cache_data(show_spinner=False)
def create_node_throughput_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create input throughput over time graph."""
    return create_node_metric_graph(
        node_metrics_list,
        title="Input Throughput Over Time by Node",
        y_label="Input Throughput (tokens/s)",
        metric_key="input_throughput",
        mode="lines+markers",
        group_by_dp=group_by_dp,
        aggregate_all=aggregate_all,
    )


@st.cache_data(show_spinner=False)
def create_cache_hit_rate_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create cache hit rate visualization."""

    def calculate_hit_rate(batch):
        if batch.get("type") == "prefill" and "new_token" in batch and "cached_token" in batch:
            new_tokens = batch["new_token"]
            cached_tokens = batch["cached_token"]
            total_tokens = new_tokens + cached_tokens
            return (cached_tokens / total_tokens) * 100 if total_tokens > 0 else None
        return None

    fig = create_node_metric_graph(
        node_metrics_list,
        title="Cache Hit Rate Over Time",
        y_label="Cache Hit Rate (%)",
        metric_key=None,
        value_extractor=calculate_hit_rate,
        mode="lines+markers",
        group_by_dp=group_by_dp,
        aggregate_all=aggregate_all,
    )
    fig.update_layout(yaxis={"range": [0, 100]})
    return fig


@st.cache_data(show_spinner=False)
def create_kv_cache_utilization_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create KV cache utilization visualization."""
    fig = create_node_metric_graph(
        node_metrics_list,
        title="KV Cache Utilization Over Time",
        y_label="Utilization (%)",
        metric_key="token_usage",
        value_extractor=lambda b: b.get("token_usage", 0) * 100,
        mode="lines+markers",
        group_by_dp=group_by_dp,
        aggregate_all=aggregate_all,
    )
    fig.update_layout(yaxis={"range": [0, 100]})
    return fig


@st.cache_data(show_spinner=False)
def create_queue_depth_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create queued requests visualization."""
    fig = create_node_metric_graph(
        node_metrics_list,
        title="Queued Requests Over Time",
        y_label="Number of Requests",
        metric_key="queue_req",
        mode="lines+markers",
        group_by_dp=group_by_dp,
        aggregate_all=aggregate_all,
    )
    fig.update_layout(hovermode="x unified")
    return fig


@st.cache_data(show_spinner=False)
def create_node_inflight_requests_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create inflight requests visualization."""
    fig = create_node_metric_graph(
        node_metrics_list,
        title="Inflight Requests Over Time",
        y_label="Number of Requests",
        metric_key="inflight_req",
        mode="lines",
        stackgroup="one",
        group_by_dp=group_by_dp,
        aggregate_all=aggregate_all,
    )
    fig.update_layout(hovermode="x unified")
    return fig


@st.cache_data(show_spinner=False)
def create_decode_running_requests_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create running requests visualization for decode nodes."""
    fig = create_node_metric_graph(
        node_metrics_list,
        title="Running Requests Over Time",
        y_label="Number of Requests",
        metric_key="running_req",
        batch_filter=lambda b: b.get("type") == "decode",
        mode="lines+markers",
        group_by_dp=group_by_dp,
        aggregate_all=aggregate_all,
    )
    fig.update_layout(hovermode="x unified")
    return fig


@st.cache_data(show_spinner=False)
def create_decode_gen_throughput_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create generation throughput visualization for decode nodes."""
    return create_node_metric_graph(
        node_metrics_list,
        title="Generation Throughput Over Time",
        y_label="Gen Throughput (tokens/s)",
        metric_key="gen_throughput",
        batch_filter=lambda b: b.get("type") == "decode",
        mode="lines+markers",
        group_by_dp=group_by_dp,
        aggregate_all=aggregate_all,
    )


@st.cache_data(show_spinner=False)
def create_decode_transfer_req_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create transfer requests visualization for decode nodes."""
    fig = create_node_metric_graph(
        node_metrics_list,
        title="Transfer Requests Over Time",
        y_label="Number of Requests",
        metric_key="transfer_req",
        batch_filter=lambda b: b.get("type") == "decode",
        mode="lines",
        stackgroup="one",
        group_by_dp=group_by_dp,
        aggregate_all=aggregate_all,
    )
    fig.update_layout(hovermode="x unified")
    return fig


@st.cache_data(show_spinner=False)
def create_decode_prealloc_req_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create prealloc requests visualization for decode nodes."""
    fig = create_node_metric_graph(
        node_metrics_list,
        title="Prealloc Requests Over Time",
        y_label="Number of Requests",
        metric_key="prealloc_req",
        batch_filter=lambda b: b.get("type") == "decode",
        mode="lines",
        stackgroup="one",
        group_by_dp=group_by_dp,
        aggregate_all=aggregate_all,
    )
    fig.update_layout(hovermode="x unified")
    return fig


@st.cache_data(show_spinner=False)
def create_decode_disagg_stacked_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create stacked area chart for disaggregation request flow."""
    metrics_config = [
        {"key": "prealloc_req", "name": "Prealloc Queue", "color": "rgba(99, 110, 250, 0.3)"},
        {"key": "transfer_req", "name": "Transfer Queue", "color": "rgba(239, 85, 59, 0.3)"},
        {"key": "running_req", "name": "Running", "color": "rgba(0, 204, 150, 0.3)"},
    ]

    return create_stacked_metric_graph(
        node_metrics_list,
        title="Disaggregation Request Flow (Stacked)",
        metrics_config=metrics_config,
        batch_filter=lambda b: b.get("type") == "decode",
        group_by_dp=group_by_dp,
        aggregate_all=aggregate_all,
    )


def main():
    # Header
    st.markdown(
        '<div class="main-header">📊 Benchmark Analysis Dashboard</div>', unsafe_allow_html=True
    )
    st.markdown("Interactive visualization and analysis of benchmark logs")

    # Sidebar - Directory selection and filters
    st.sidebar.header("Configuration")

    # Directory input
    default_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = st.sidebar.text_input(
        "Logs Directory Path",
        value=default_dir,
        help="Path to the directory containing benchmark log folders",
    )

    if not os.path.exists(logs_dir):
        st.error(f"Directory not found: {logs_dir}")
        return

    # Load data
    with st.spinner("Loading benchmark data..."):
        runs = load_data(logs_dir)

    if not runs:
        st.warning("No benchmark runs found in the specified directory.")
        st.info("Make sure the directory contains subdirectories with benchmark results.")
        return

    # Run selection with formatted labels
    st.sidebar.header("Run Selection")

    # Always sort by date, newest first
    sorted_runs = sorted(runs.copy(), key=lambda r: r.get("run_date", ""), reverse=True)

    # Add filtering options
    st.sidebar.subheader("Filters")

    # 1. Date Range Filter
    with st.sidebar.expander("📅 Date Range", expanded=False):
        # Get min/max dates from runs
        dates_with_data = [
            r.get("run_date")
            for r in sorted_runs
            if r.get("run_date") and r.get("run_date") != "N/A"
        ]

        if dates_with_data:
            from datetime import datetime, timedelta

            date_objects = []
            for date_str in dates_with_data:
                try:
                    date_objects.append(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S"))
                except Exception:
                    pass

            if date_objects:
                min_date = min(date_objects).date()
                max_date = max(date_objects).date()

                date_filter_option = st.radio(
                    "Select date range",
                    options=["All time", "Last 7 days", "Last 30 days", "Custom range"],
                    key="date_filter_option",
                )

                if date_filter_option == "Last 7 days":
                    filter_start_date = max_date - timedelta(days=7)
                    filter_end_date = max_date
                elif date_filter_option == "Last 30 days":
                    filter_start_date = max_date - timedelta(days=30)
                    filter_end_date = max_date
                elif date_filter_option == "Custom range":
                    filter_start_date = st.date_input(
                        "From",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="date_from",
                    )
                    filter_end_date = st.date_input(
                        "To", value=max_date, min_value=min_date, max_value=max_date, key="date_to"
                    )
                else:  # All time
                    filter_start_date = None
                    filter_end_date = None

                # Apply date filter
                if filter_start_date and filter_end_date:
                    filtered_by_date = []
                    for run in sorted_runs:
                        run_date_str = run.get("run_date")
                        if run_date_str and run_date_str != "N/A":
                            try:
                                run_date = datetime.strptime(
                                    run_date_str, "%Y-%m-%d %H:%M:%S"
                                ).date()
                                if filter_start_date <= run_date <= filter_end_date:
                                    filtered_by_date.append(run)
                            except Exception:
                                pass
                    sorted_runs = filtered_by_date
        else:
            st.caption("No date information available")

    # 2. Topology Filter
    with st.sidebar.expander("🔧 Topology", expanded=False):
        # Extract unique topologies
        topologies = set()
        for run in sorted_runs:
            prefill_dp = run.get("prefill_dp", "?")
            decode_dp = run.get("decode_dp", "?")
            topology = f"{prefill_dp}P/{decode_dp}D"
            if topology != "?P/?D":
                topologies.add(topology)

        if topologies:
            topology_options = sorted(topologies)
            selected_topologies = st.multiselect(
                "Select topologies",
                options=topology_options,
                default=topology_options,
                key="topology_filter",
            )

            # Apply topology filter
            if selected_topologies:
                filtered_by_topology = []
                for run in sorted_runs:
                    topology = f"{run.get('prefill_dp', '?')}P/{run.get('decode_dp', '?')}D"
                    if topology in selected_topologies:
                        filtered_by_topology.append(run)
                sorted_runs = filtered_by_topology
        else:
            st.caption("No topology information available")

    # 3. ISL/OSL Filter
    with st.sidebar.expander("📊 ISL/OSL", expanded=False):
        # Extract unique ISL and OSL values
        isl_values = set()
        osl_values = set()
        for run in sorted_runs:
            isl = run.get("isl")
            osl = run.get("osl")
            if isl and isl != "N/A" and isl != "?":
                isl_values.add(isl)
            if osl and osl != "N/A" and osl != "?":
                osl_values.add(osl)

        if isl_values:
            isl_options = sorted(isl_values)
            selected_isl = st.multiselect(
                "Input Sequence Length (ISL)",
                options=isl_options,
                default=isl_options,
                key="isl_filter",
            )
        else:
            selected_isl = None
            st.caption("No ISL information available")

        if osl_values:
            osl_options = sorted(osl_values)
            selected_osl = st.multiselect(
                "Output Sequence Length (OSL)",
                options=osl_options,
                default=osl_options,
                key="osl_filter",
            )
        else:
            selected_osl = None
            st.caption("No OSL information available")

        # Apply ISL/OSL filter
        if selected_isl or selected_osl:
            filtered_by_isl_osl = []
            for run in sorted_runs:
                isl_match = (not selected_isl) or (run.get("isl") in selected_isl)
                osl_match = (not selected_osl) or (run.get("osl") in selected_osl)
                if isl_match and osl_match:
                    filtered_by_isl_osl.append(run)
            sorted_runs = filtered_by_isl_osl

    # 4. Container Filter
    with st.sidebar.expander("🐳 Container", expanded=False):
        # Extract unique containers
        container_values = set()
        for run in sorted_runs:
            container = run.get("container")
            if container and container != "N/A":
                container_values.add(container)

        if container_values:
            container_options = sorted(container_values)
            selected_containers = st.multiselect(
                "Select containers",
                options=container_options,
                default=container_options,
                key="container_filter",
            )

            # Apply container filter
            if selected_containers:
                filtered_by_container = []
                for run in sorted_runs:
                    container = run.get("container")
                    if container in selected_containers or (
                        not container and "N/A" in selected_containers
                    ):
                        filtered_by_container.append(run)
                sorted_runs = filtered_by_container
        else:
            st.caption("No container information available")

    # Show filter results
    st.sidebar.caption(f"✅ {len(sorted_runs)} runs match filters")

    # Create formatted labels and mapping
    run_labels = []
    label_to_run = {}

    for run in sorted_runs:
        job_id = run.get("slurm_job_id", "Unknown")
        date = run.get("run_date", "N/A")

        # Parse date for shorter display
        if date != "N/A":
            try:
                from datetime import datetime

                date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                date_short = date_obj.strftime("%b %d").replace(" 0", " ")  # "Nov 4"
            except Exception:
                date_short = date.split()[0]  # Fallback to YYYY-MM-DD
        else:
            date_short = "No date"

        # Extract job number
        job_num = job_id.split("_")[0] if "_" in job_id else job_id

        # Create readable label
        topology = f"{run.get('prefill_dp', '?')}P/{run.get('decode_dp', '?')}D"
        isl = run.get("isl", "?")
        label = f"[{date_short}] Job {job_num} - {topology} (ISL {isl})"

        run_labels.append(label)
        label_to_run[label] = run

    # Multiselect with formatted labels
    selected_labels = st.sidebar.multiselect(
        "Select runs to compare",
        options=run_labels,
        default=run_labels[: min(3, len(run_labels))],  # Select first 3 by default
        help="Select one or more runs to visualize",
    )

    if not selected_labels:
        st.warning("Please select at least one run to visualize.")
        return

    # Filter runs based on selected labels
    filtered_runs = [label_to_run[label] for label in selected_labels]

    # Extract run IDs for compatibility with existing graph functions
    selected_runs = [run.get("slurm_job_id", "Unknown") for run in filtered_runs]

    # Get dataframe - use helper function to convert dicts
    df = _runs_to_dataframe(filtered_runs)

    # Filters
    st.sidebar.header("Filters")

    # Concurrency range filter
    min_concurrency = int(df["Concurrency"].min())
    max_concurrency = int(df["Concurrency"].max())

    if min_concurrency < max_concurrency:
        concurrency_range = st.sidebar.slider(
            "Concurrency Range",
            min_value=min_concurrency,
            max_value=max_concurrency,
            value=(min_concurrency, max_concurrency),
        )
        df = df[
            (df["Concurrency"] >= concurrency_range[0])
            & (df["Concurrency"] <= concurrency_range[1])
        ]

    # Pareto options
    st.sidebar.header("Pareto Graph Options")
    y_axis_metric = st.sidebar.radio(
        "Y-axis metric",
        options=["Output TPS/GPU", "Total TPS/GPU"],
        index=0,
        help="Choose between decode throughput per GPU or total throughput per GPU (input + output)",
    )
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

    # Summary metrics
    st.header("Summary")

    # Show unique containers
    containers = [run.get("container") for run in filtered_runs if run.get("container")]
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
        unique_profilers = df["Profiler"].nunique()
        st.metric("Profilers", unique_profilers)

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📈 Pareto Graph",
            "⏱️ Latency Analysis",
            "🖥️ Node Metrics",
            "⚙️ Configuration",
            "🔬 Run Comparison",
        ]
    )

    with tab1:
        st.subheader("Pareto Frontier Analysis")
        
        if y_axis_metric == "Total TPS/GPU":
            st.markdown("""
            This graph shows the trade-off between **Total TPS/GPU** (input + output tokens/s per GPU) and
            **Output TPS/User** (throughput per user).
            """)
        else:
            st.markdown("""
            This graph shows the trade-off between **Output TPS/GPU** (decode tokens/s per GPU) and
            **Output TPS/User** (throughput per user).
            """)

        pareto_fig = create_pareto_graph(
            df, selected_runs, show_cutoff, cutoff_value, show_frontier, y_axis_metric
        )
        pareto_fig.update_xaxes(showgrid=True)
        pareto_fig.update_yaxes(showgrid=True)

        st.plotly_chart(pareto_fig, width="stretch", key="pareto_main")

        # Debug info for frontier
        if show_frontier:
            frontier_points = calculate_pareto_frontier(df, y_axis_metric)
            st.caption(
                f"🔍 Debug: Frontier has {len(frontier_points)} points across {len(df)} total data points"
            )

            # Show which points are on the frontier
            if len(frontier_points) > 0:
                with st.expander("View Frontier Points Details"):
                    frontier_df = pd.DataFrame(
                        frontier_points, columns=["Output TPS/User", "Output TPS/GPU"]
                    )
                    st.dataframe(frontier_df, width="stretch")

        # Add data export button below the graph
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.download_button(
                label="📥 Download Data as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="benchmark_data.csv",
                mime="text/csv",
                width="stretch",
            )

        # Metric calculation documentation
        st.divider()
        st.markdown("### 📊 Metric Calculations")
        st.markdown("""
        **How each metric is calculated:**

        **Output TPS/GPU** (Throughput Efficiency):
        """)
        st.latex(
            r"\text{Output TPS/GPU} = \frac{\text{Total Output Throughput (tokens/s)}}{\text{Total Number of GPUs}}"
        )
        st.markdown("""
        *This measures how efficiently each GPU is being utilized for token generation.*

        **Output TPS/User** (Per-User Generation Rate):
        """)
        st.latex(r"\text{Output TPS/User} = \frac{1000}{\text{Mean TPOT (ms)}}")
        st.markdown("""
        *Where TPOT (Time Per Output Token) is the average time between consecutive output tokens.
        This represents the actual token generation rate experienced by each user, independent of concurrency.*
        """)

    with tab2:
        st.subheader("Latency Analysis")

        if len(selected_runs) == 0:
            st.warning("Please select at least one run from the sidebar.")
        else:
            # TTFT Graph
            st.markdown("### Time to First Token (TTFT)")
            st.markdown("""
            **TTFT** measures the time from request submission to receiving the first output token.
            This is critical for perceived responsiveness.
            """)
            ttft_fig = create_latency_vs_concurrency_graph(
                df,
                selected_runs,
                metric_name="TTFT",
                metric_col="Mean TTFT (ms)",
                y_label="Mean TTFT (ms)",
            )
            st.plotly_chart(ttft_fig, width="stretch", key="latency_ttft")

            # TPOT Graph
            st.markdown("### Time Per Output Token (TPOT)")
            st.markdown("""
            **TPOT** measures the time between consecutive output tokens during generation.
            Lower TPOT means faster streaming and better user experience.
            """)
            tpot_fig = create_latency_vs_concurrency_graph(
                df,
                selected_runs,
                metric_name="TPOT",
                metric_col="Mean TPOT (ms)",
                y_label="Mean TPOT (ms)",
            )
            st.plotly_chart(tpot_fig, width="stretch", key="latency_tpot")

            # ITL Graph
            st.markdown("### Inter-Token Latency (ITL)")
            st.markdown("""
            **ITL** measures the interval between tokens during generation.
            Similar to TPOT but may include queueing delays.
            """)
            itl_fig = create_latency_vs_concurrency_graph(
                df,
                selected_runs,
                metric_name="ITL",
                metric_col="Mean ITL (ms)",
                y_label="Mean ITL (ms)",
            )
            st.plotly_chart(itl_fig, width="stretch", key="latency_itl")

            # Summary statistics
            st.divider()
            st.markdown("### 📊 Latency Summary Statistics")

            summary_data = []
            for run_id in selected_runs:
                run_data = df[df["Run ID"] == run_id]
                if len(run_data) > 0:
                    summary_data.append(
                        {
                            "Run ID": run_id,
                            "Min TTFT (ms)": run_data["Mean TTFT (ms)"].min()
                            if "Mean TTFT (ms)" in run_data.columns
                            else "N/A",
                            "Max TTFT (ms)": run_data["Mean TTFT (ms)"].max()
                            if "Mean TTFT (ms)" in run_data.columns
                            else "N/A",
                            "Min TPOT (ms)": run_data["Mean TPOT (ms)"].min()
                            if "Mean TPOT (ms)" in run_data.columns
                            else "N/A",
                            "Max TPOT (ms)": run_data["Mean TPOT (ms)"].max()
                            if "Mean TPOT (ms)" in run_data.columns
                            else "N/A",
                            "Min ITL (ms)": run_data["Mean ITL (ms)"].min()
                            if "Mean ITL (ms)" in run_data.columns
                            else "N/A",
                            "Max ITL (ms)": run_data["Mean ITL (ms)"].max()
                            if "Mean ITL (ms)" in run_data.columns
                            else "N/A",
                        }
                    )

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, width="stretch")

    with tab3:
        st.subheader("Node-Level Metrics")
        st.markdown("""
        Runtime metrics extracted from log files, split by prefill and decode nodes.
        """)

        # Parse log files for all selected runs (cached)
        all_node_metrics = []
        with st.spinner(f"Parsing logs for {len(filtered_runs)} run(s)..."):
            for run in filtered_runs:
                run_path = run.get("path", "")
                run_id = run.get("slurm_job_id", "Unknown")
                if run_path and os.path.exists(run_path):
                    node_metrics = load_node_metrics(run_path)
                    # Add run_id to each node for identification in multi-run comparisons
                    for node_data in node_metrics:
                        node_data["run_id"] = run_id
                    all_node_metrics.extend(node_metrics)

        if not all_node_metrics:
            st.warning("No log files (.err) found for the selected runs.")
            st.info(
                "Node metrics are extracted from files like `*_prefill_*.err` and `*_decode_*.err`"
            )
        else:
            # Split by prefill vs decode
            prefill_nodes = [
                n for n in all_node_metrics if n["node_info"]["worker_type"] == "prefill"
            ]
            decode_nodes = [
                n for n in all_node_metrics if n["node_info"]["worker_type"] == "decode"
            ]

            st.caption(
                f"📊 Found {len(prefill_nodes)} prefill nodes, {len(decode_nodes)} decode nodes"
            )

            # Add toggle for grouping
            aggregation_mode = st.radio(
                "📊 Node Aggregation",
                options=[
                    "Show individual nodes",
                    "Group by DP rank (average per DP)",
                    "Aggregate all nodes (single averaged line)",
                ],
                index=2,  # Default to aggregate all
                horizontal=True,
                help="Control how node metrics are displayed: individual lines, grouped by DP rank, or fully aggregated across all nodes.",
            )

            # Pre-aggregate nodes ONCE to avoid recomputing for each graph
            from srtslurm.visualizations import aggregate_all_nodes, group_nodes_by_dp

            if aggregation_mode == "Aggregate all nodes (single averaged line)":
                with st.spinner("Aggregating nodes..."):
                    prefill_nodes = aggregate_all_nodes(prefill_nodes)
                    decode_nodes = aggregate_all_nodes(decode_nodes)
                group_by_dp = False
                aggregate_all = False  # Already aggregated
            elif aggregation_mode == "Group by DP rank (average per DP)":
                with st.spinner("Grouping by DP..."):
                    prefill_nodes = group_nodes_by_dp(prefill_nodes)
                    decode_nodes = group_nodes_by_dp(decode_nodes)
                group_by_dp = False
                aggregate_all = False  # Already grouped
            else:
                # Individual nodes - no preprocessing
                group_by_dp = False
                aggregate_all = False

            # Prefill Metrics Section
            if prefill_nodes:
                st.markdown("### 📤 Prefill Node Metrics")

                # Vertically stack graphs for better horizontal stretching
                throughput_fig = create_node_throughput_graph(
                    prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
                )
                throughput_fig.update_xaxes(showgrid=True)
                throughput_fig.update_yaxes(showgrid=True)
                st.plotly_chart(throughput_fig, width="stretch", key="prefill_throughput")
                st.caption(
                    "Shows prefill throughput in tokens/s - measures how fast the system processes input prompts"
                )

                inflight_fig = create_node_inflight_requests_graph(
                    prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
                )
                inflight_fig.update_xaxes(showgrid=True)
                inflight_fig.update_yaxes(showgrid=True)
                st.plotly_chart(inflight_fig, width="stretch", key="prefill_inflight")
                st.caption(
                    "Requests that have been sent to decode workers in PD disaggregation mode"
                )

                # Cache hit rate graph hidden for now
                # cache_fig = create_cache_hit_rate_graph(
                #     prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
                # )
                # cache_fig.update_xaxes(showgrid=True)
                # cache_fig.update_yaxes(showgrid=True)
                # st.plotly_chart(cache_fig, width="stretch", key="prefill_cache_hit")
                # st.caption(
                #     "Percentage of tokens found in prefix cache - higher values indicate better cache reuse and reduced compute"
                # )

                kv_fig = create_kv_cache_utilization_graph(
                    prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
                )
                kv_fig.update_xaxes(showgrid=True)
                kv_fig.update_yaxes(showgrid=True)
                st.plotly_chart(kv_fig, width="stretch", key="prefill_kv_util")
                st.caption(
                    "Percentage of KV cache memory currently in use - helps tune max-total-tokens and identify memory pressure"
                )

                queue_fig = create_queue_depth_graph(
                    prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
                )
                queue_fig.update_layout(title="PREFILL Queued Requests")
                queue_fig.update_xaxes(showgrid=True)
                queue_fig.update_yaxes(showgrid=True)
                st.plotly_chart(queue_fig, width="stretch", key="prefill_queue_v2")
                st.caption(
                    "Prefill requests waiting in queue - growing queue indicates backpressure or overload"
                )

            # Decode Metrics Section
            if decode_nodes:
                st.divider()
                st.markdown("### 📥 Decode Node Metrics")

                # Debug: Check if decode nodes have batch data
                has_data = any(node_data["prefill_batches"] for node_data in decode_nodes)
                if not has_data:
                    st.warning(
                        "⚠️ No batch metrics found for decode nodes. Decode nodes may not log batch-level metrics in the current setup."
                    )
                else:
                    running_fig = create_decode_running_requests_graph(
                        decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
                    )
                    running_fig.update_xaxes(showgrid=True)
                    running_fig.update_yaxes(showgrid=True)
                    st.plotly_chart(running_fig, width="stretch", key="decode_running")
                    st.caption(
                        "Number of requests currently being decoded and generating output tokens"
                    )

                    gen_fig = create_decode_gen_throughput_graph(
                        decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
                    )
                    gen_fig.update_xaxes(showgrid=True)
                    gen_fig.update_yaxes(showgrid=True)
                    st.plotly_chart(gen_fig, width="stretch", key="decode_gen_throughput")
                    st.caption(
                        "Output token generation rate in tokens/s - measures decode performance"
                    )

                    kv_decode_fig = create_kv_cache_utilization_graph(
                        decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
                    )
                    kv_decode_fig.update_xaxes(showgrid=True)
                    kv_decode_fig.update_yaxes(showgrid=True)
                    st.plotly_chart(kv_decode_fig, width="stretch", key="decode_kv_util")
                    st.caption(
                        "Total KV cache tokens in use across all running requests - low indicates underutilization, high indicates risk of OOM"
                    )

                    queue_decode_fig = create_queue_depth_graph(
                        decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
                    )
                    queue_decode_fig.update_layout(title="DECODE Queued Requests")
                    queue_decode_fig.update_xaxes(showgrid=True)
                    queue_decode_fig.update_yaxes(showgrid=True)
                    st.plotly_chart(queue_decode_fig, width="stretch", key="decode_queue_v2")
                    st.caption(
                        "Decode requests waiting in queue - indicates decode capacity constraints"
                    )

                    # Rate Matching Graph
                    st.divider()
                    st.markdown("#### Rate Match")
                    st.caption(
                        "Compare prefill input rate vs decode generation rate to verify proper node ratio"
                    )

                    # Create rate match graph
                    rate_fig = go.Figure()

                    # Get prefill input throughput over time
                    if prefill_nodes:
                        for p_node in prefill_nodes:
                            timestamps = []
                            input_tps = []

                            for batch in p_node["prefill_batches"]:
                                if batch.get("input_throughput") is not None:
                                    ts = batch.get("timestamp", "")
                                    if ts:
                                        timestamps.append(ts)
                                        input_tps.append(batch["input_throughput"])

                            if timestamps:
                                # Convert timestamps to elapsed seconds from first timestamp
                                from datetime import datetime

                                first_time = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
                                elapsed = [
                                    (
                                        datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - first_time
                                    ).total_seconds()
                                    for ts in timestamps
                                ]

                                rate_fig.add_trace(
                                    go.Scatter(
                                        x=elapsed,
                                        y=input_tps,
                                        mode="lines+markers",
                                        name="Prefill Input (tok/s)",
                                        line={"color": "orange", "width": 2},
                                    )
                                )

                    # Get decode gen throughput over time (sum across all decode nodes)
                    if decode_nodes:
                        # Collect all decode batches with timestamps
                        all_decode_batches = {}
                        for d_node in decode_nodes:
                            for batch in d_node["prefill_batches"]:
                                if (
                                    batch.get("gen_throughput") is not None
                                    and batch.get("gen_throughput") > 0
                                ):
                                    ts = batch.get("timestamp", "")
                                    if ts:
                                        if ts not in all_decode_batches:
                                            all_decode_batches[ts] = []
                                        all_decode_batches[ts].append(batch["gen_throughput"])

                        # Average across nodes at each timestamp, then multiply by num decode nodes
                        timestamps = []
                        total_gen_tps = []
                        num_decode = len(decode_nodes)

                        for ts in sorted(all_decode_batches.keys()):
                            avg_gen = sum(all_decode_batches[ts]) / len(all_decode_batches[ts])
                            total_gen = avg_gen * num_decode  # Scale by number of decode nodes
                            timestamps.append(ts)
                            total_gen_tps.append(total_gen)

                        if timestamps:
                            # Convert to elapsed seconds
                            from datetime import datetime

                            first_time = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
                            elapsed = [
                                (
                                    datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - first_time
                                ).total_seconds()
                                for ts in timestamps
                            ]

                            rate_fig.add_trace(
                                go.Scatter(
                                    x=elapsed,
                                    y=total_gen_tps,
                                    mode="lines+markers",
                                    name=f"Decode Gen (tok/s) × {num_decode} nodes",
                                    line=dict(color="green", width=2),
                                )
                            )

                    rate_fig.update_layout(
                        title="Rate Match: Prefill Input vs Decode Generation",
                        xaxis_title="Time Elapsed (seconds)",
                        yaxis_title="Throughput (tokens/s)",
                        hovermode="x unified",
                        height=500,
                    )
                    rate_fig.update_xaxes(showgrid=True)
                    rate_fig.update_yaxes(showgrid=True)

                    st.plotly_chart(rate_fig, width="stretch", key="rate_match")
                    st.caption(
                        f"Rate matched when lines align. Prefill: {len(prefill_nodes)} node(s), Decode: {len(decode_nodes)} node(s)"
                    )

                    # Disaggregation metrics with toggle
                    st.divider()
                    st.markdown("#### Disaggregation Metrics")

                    disagg_view = st.radio(
                        "View mode",
                        options=["Stacked (Combined)", "Separate Graphs"],
                        index=0,
                        horizontal=True,
                        help="Stacked view shows request flow through stages. Separate graphs show individual metrics.",
                    )

                    if disagg_view == "Stacked (Combined)":
                        stacked_fig = create_decode_disagg_stacked_graph(
                            decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
                        )
                        stacked_fig.update_xaxes(showgrid=True)
                        stacked_fig.update_yaxes(showgrid=True)
                        st.plotly_chart(stacked_fig, width="stretch", key="decode_disagg_stacked")
                        st.caption(
                            "Shows the request flow funnel: Prealloc Queue → Transfer Queue → Running requests in PD disaggregation"
                        )
                    else:
                        transfer_fig = create_decode_transfer_req_graph(
                            decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
                        )
                        transfer_fig.update_xaxes(showgrid=True)
                        transfer_fig.update_yaxes(showgrid=True)
                        st.plotly_chart(transfer_fig, width="stretch", key="decode_transfer")
                        st.caption(
                            "Requests waiting for KV cache transfer from prefill to decode workers in PD disaggregation mode"
                        )

                        prealloc_fig = create_decode_prealloc_req_graph(
                            decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
                        )
                        prealloc_fig.update_xaxes(showgrid=True)
                        prealloc_fig.update_yaxes(showgrid=True)
                        st.plotly_chart(prealloc_fig, width="stretch", key="decode_prealloc")
                        st.caption(
                            "Requests in pre-allocation queue for PD disaggregation - waiting for memory allocation on decode workers"
                        )

    with tab4:
        st.subheader("Run Configuration Details")

        for run in filtered_runs:
            run_id = run.get("slurm_job_id", "Unknown")
            run_path = run.get("path", "")
            run_date = run.get("run_date", None)

            # Add date to expander title if available
            expander_title = f"🔧 {run_id}"
            if run_date and run_date != "N/A":
                expander_title += f" ({run_date})"

            with st.expander(expander_title, expanded=len(filtered_runs) == 1):
                if not run_path or not os.path.exists(run_path):
                    st.warning("Configuration files not found for this run.")
                    continue

                # Get structured config data
                config_data = format_config_for_display(run_path)

                if "error" in config_data:
                    st.error(config_data["error"])
                    continue

                # Compact overview at top
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Nodes", config_data["summary"]["num_nodes"])
                with col2:
                    st.metric("GPU", config_data["summary"]["gpu_type"])
                with col3:
                    st.metric("ISL/OSL", f"{run.get('isl', 'N/A')}/{run.get('osl', 'N/A')}")
                with col4:
                    st.metric("Profiler", run.get("profiler_type", "N/A"))

                st.caption(f"Model: {config_data['summary']['model']}")
                st.divider()

                # Use tabs for cleaner organization
                config_tab1, config_tab2, config_tab3 = st.tabs(
                    ["📋 Topology", "⚙️ Node Config", "🌍 Environment"]
                )

                # Get all configs for use in tabs
                all_configs = get_all_configs(run_path)

                with config_tab1:
                    # Topology - compact summary
                    parsed_data = parse_command_line_from_err(run_path)
                    physical_nodes = parsed_data.get("services", {})

                    if physical_nodes:
                        # Count by service type
                        service_counts = {}
                        for services in physical_nodes.values():
                            for svc in set(services):
                                service_counts[svc] = service_counts.get(svc, 0) + 1

                        # Display summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prefill Nodes", service_counts.get("prefill", 0))
                        with col2:
                            st.metric("Decode Nodes", service_counts.get("decode", 0))
                        with col3:
                            st.metric("Frontend Nodes", service_counts.get("frontend", 0))

                        # Compact node grid (collapsed by default)
                        with st.expander(
                            f"View all {len(physical_nodes)} physical nodes", expanded=False
                        ):
                            cols_per_row = 4
                            node_items = sorted(physical_nodes.items())
                            for i in range(0, len(node_items), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j, (phys_node, service_types) in enumerate(
                                    node_items[i : i + cols_per_row]
                                ):
                                    with cols[j]:
                                        st.caption(f"**{phys_node}**")
                                        for svc in sorted(set(service_types)):
                                            emoji = {
                                                "prefill": "📤",
                                                "decode": "📥",
                                                "frontend": "🌐",
                                                "nginx": "🌐",
                                            }.get(svc, "⚙️")
                                            st.caption(f"{emoji} {svc}")
                    else:
                        st.info("No topology information available")

                with config_tab2:
                    # Node Configuration - dropdown to select specific node
                    if all_configs:
                        # Create node selection dropdown
                        node_names = [
                            config.get("filename", f"Node {i}")
                            .replace("_config.json", "")
                            .replace("watchtower-aqua-", "")
                            .replace("watchtower-navy-", "")
                            for i, config in enumerate(all_configs)
                        ]

                        # Group by type
                        prefill_nodes = [
                            (i, name)
                            for i, name in enumerate(node_names)
                            if "prefill" in name.lower()
                        ]
                        decode_nodes = [
                            (i, name)
                            for i, name in enumerate(node_names)
                            if "decode" in name.lower()
                        ]
                        other_nodes = [
                            (i, name)
                            for i, name in enumerate(node_names)
                            if "prefill" not in name.lower() and "decode" not in name.lower()
                        ]

                        # Create categorized options
                        node_options = []
                        if prefill_nodes:
                            node_options.extend([f"📤 {name}" for _, name in prefill_nodes])
                        if decode_nodes:
                            node_options.extend([f"📥 {name}" for _, name in decode_nodes])
                        if other_nodes:
                            node_options.extend([f"🖥️ {name}" for _, name in other_nodes])

                        # Map back to indices
                        option_to_idx = {}
                        all_indexed = prefill_nodes + decode_nodes + other_nodes
                        for i, (idx, _) in enumerate(all_indexed):
                            option_to_idx[node_options[i]] = idx

                        selected_option = st.selectbox(
                            "Select node",
                            options=node_options,
                            key=f"config_node_{run_id}",
                        )

                        selected_idx = option_to_idx[selected_option]
                        selected_config = all_configs[selected_idx]

                        # Show command line args (actual flags passed)
                        cmd_args = get_command_line_args(selected_config)

                        if cmd_args:
                            cmd_dict = parse_command_line_to_dict(cmd_args)

                            st.markdown(f"**Command Line Arguments** ({len(cmd_dict)} flags)")
                            st.caption("Actual flags passed on command line for this node")

                            # Display all flags in 3 columns
                            num_items = len(cmd_dict)
                            items_per_col = (num_items + 2) // 3

                            col1, col2, col3 = st.columns(3)

                            for idx, (key, value) in enumerate(sorted(cmd_dict.items())):
                                col_idx = idx // items_per_col
                                display_val = (
                                    str(value)[:60] + "..." if len(str(value)) > 60 else value
                                )

                                if col_idx == 0:
                                    with col1:
                                        st.caption(f"`{key}`: **{display_val}**")
                                elif col_idx == 1:
                                    with col2:
                                        st.caption(f"`{key}`: **{display_val}**")
                                else:
                                    with col3:
                                        st.caption(f"`{key}`: **{display_val}**")
                        else:
                            st.info("No command line args found")
                    else:
                        st.info("No config files found")

                with config_tab3:
                    # Environment Variables
                    if all_configs:
                        # Use same node selector
                        node_names = [
                            config.get("filename", f"Node {i}")
                            .replace("_config.json", "")
                            .replace("watchtower-aqua-", "")
                            .replace("watchtower-navy-", "")
                            for i, config in enumerate(all_configs)
                        ]

                        # Simple select (no emojis this time)
                        selected_name = st.selectbox(
                            "Select node",
                            options=node_names,
                            key=f"env_node_{run_id}",
                        )

                        selected_config = all_configs[node_names.index(selected_name)]
                        env_vars = get_environment_variables(selected_config)

                        if env_vars:
                            for category, vars_dict in env_vars.items():
                                with st.expander(
                                    f"{category} ({len(vars_dict)} vars)",
                                    expanded=category in ["NCCL", "SGLANG"],
                                ):
                                    for key, value in sorted(vars_dict.items()):
                                        st.caption(f"`{key}`: {value}")
                        else:
                            st.info("No environment variables found")
                    else:
                        st.info("No config files found")

    with tab5:
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
                "Sort by", options=["Newest First", "Oldest First", "Job ID"], key="comparison_sort"
            )

        # Sort filtered_runs based on selection
        sorted_runs = filtered_runs.copy()
        if sort_by == "Newest First":
            sorted_runs = sorted(sorted_runs, key=lambda r: r.get("run_date", ""), reverse=True)
        elif sort_by == "Oldest First":
            sorted_runs = sorted(sorted_runs, key=lambda r: r.get("run_date", ""))
        elif sort_by == "Job ID":
            sorted_runs = sorted(sorted_runs, key=lambda r: r.get("slurm_job_id", ""))

        # Create formatted labels and mapping
        run_options = []
        run_map = {}

        for run in sorted_runs:
            job_id = run.get("slurm_job_id", "Unknown")
            date = run.get("run_date", "N/A")

            # Parse date for shorter display
            if date != "N/A":
                # Format: "2025-11-04 23:18:43" -> "Nov 4"
                try:
                    from datetime import datetime

                    date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                    date_short = date_obj.strftime("%b %d").replace(
                        " 0", " "
                    )  # "Nov 4" (remove leading zero)
                except Exception:
                    date_short = date.split()[0]  # Fallback to YYYY-MM-DD
            else:
                date_short = "No date"

            # Extract job number from ID like "3320_1P_4D_20251104_231843"
            job_num = job_id.split("_")[0] if "_" in job_id else job_id

            # Create readable label
            topology = f"{run.get('prefill_dp', '?')}P/{run.get('decode_dp', '?')}D"
            isl = run.get("isl", "?")
            label = f"[{date_short}] Job {job_num} - {topology} (ISL {isl})"

            run_options.append(label)
            run_map[label] = run

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Run A (Baseline)**")
            run_a_label = st.selectbox(
                "Select baseline run", options=run_options, key="run_a_select"
            )
            # Get the full run data from mapping
            run_a = run_map.get(run_a_label)

            if run_a:
                st.caption(f"📅 {run_a.get('run_date', 'N/A')}")
                st.caption(f"📊 ISL/OSL: {run_a.get('isl', 'N/A')}/{run_a.get('osl', 'N/A')}")
                st.caption(
                    f"🖥️ Topology: {run_a.get('prefill_dp', '?')}P/{run_a.get('decode_dp', '?')}D"
                )
                total_gpus_a = run_a.get("prefill_tp", 0) * run_a.get("prefill_dp", 0) + run_a.get(
                    "decode_tp", 0
                ) * run_a.get("decode_dp", 0)
                st.caption(f"🎯 Total GPUs: {total_gpus_a}")

        with col2:
            st.markdown("**Run B (Comparison)**")
            run_b_label = st.selectbox(
                "Select comparison run", options=run_options, key="run_b_select"
            )
            # Get the full run data from mapping
            run_b = run_map.get(run_b_label)

            if run_b:
                st.caption(f"📅 {run_b.get('run_date', 'N/A')}")
                st.caption(f"📊 ISL/OSL: {run_b.get('isl', 'N/A')}/{run_b.get('osl', 'N/A')}")
                st.caption(
                    f"🖥️ Topology: {run_b.get('prefill_dp', '?')}P/{run_b.get('decode_dp', '?')}D"
                )
                total_gpus_b = run_b.get("prefill_tp", 0) * run_b.get("prefill_dp", 0) + run_b.get(
                    "decode_tp", 0
                ) * run_b.get("decode_dp", 0)
                st.caption(f"🎯 Total GPUs: {total_gpus_b}")

        # Validation
        if run_a_label == run_b_label:
            st.warning("⚠️ Please select two different runs to compare.")
        elif run_a and run_b:
            st.divider()

            # Get config data for both runs
            run_a_path = run_a.get("path", "")
            run_b_path = run_b.get("path", "")

            configs_a = get_all_configs(run_a_path)
            configs_b = get_all_configs(run_b_path)

            if not configs_a or not configs_b:
                st.error("Could not load configuration files for one or both runs.")
            else:
                config_a = configs_a[0]
                config_b = configs_b[0]

                # Perform comparison
                config_comparison = compare_configs(config_a, config_b)
                metrics_comparison = compare_metrics(run_a, run_b)
                scorecard = calculate_summary_scorecard(metrics_comparison)

                # Display Configuration Comparison
                with st.expander("📋 Configuration Differences", expanded=True):
                    st.markdown("#### Topology Summary")

                    topo_df = pd.DataFrame(
                        {
                            "Parameter": [
                                "Prefill TP",
                                "Prefill DP",
                                "Model",
                                "Context Length",
                                "Container",
                            ],
                            "Run A": [
                                str(config_comparison["topology_summary"]["prefill_tp"][0]),
                                str(config_comparison["topology_summary"]["prefill_dp"][0]),
                                str(config_comparison["topology_summary"]["model"][0]),
                                str(config_comparison["topology_summary"]["context_length"][0]),
                                str(run_a.get("container", "N/A")),
                            ],
                            "Run B": [
                                str(config_comparison["topology_summary"]["prefill_tp"][1]),
                                str(config_comparison["topology_summary"]["prefill_dp"][1]),
                                str(config_comparison["topology_summary"]["model"][1]),
                                str(config_comparison["topology_summary"]["context_length"][1]),
                                str(run_b.get("container", "N/A")),
                            ],
                        }
                    )
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
                            # Convert all values to strings to avoid Arrow type issues
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

                    # Format the dataframe for display
                    display_df = metrics_comparison.copy()

                    # Add improvement indicator
                    display_df["Status"] = display_df["Improved"].apply(
                        lambda x: "✅" if x else "⚠️"
                    )

                    # Select and reorder columns
                    display_df = display_df[
                        ["Concurrency", "Metric", "Run A", "Run B", "Delta", "% Change", "Status"]
                    ]

                    # Format numeric columns
                    st.dataframe(
                        display_df.style.format(
                            {
                                "Run A": "{:.2f}",
                                "Run B": "{:.2f}",
                                "Delta": "{:.2f}",
                                "% Change": "{:.1f}%",
                            }
                        ).map(
                            lambda x: "background-color: #d4edda"
                            if x == "✅"
                            else ("background-color: #f8d7da" if x == "⚠️" else ""),
                            subset=["Status"],
                        ),
                        width="stretch",
                        height=400,
                    )

                    # Visual Comparison Graphs
                    st.divider()
                    st.markdown("### 📈 Visual Comparison")

                    # Create sub-tabs for different visualizations
                    viz_tab1, viz_tab2 = st.tabs(["Side-by-Side Comparison", "Delta Graphs"])

                    with viz_tab1:
                        st.markdown("#### Overlay Comparison")
                        # Use existing pareto graph with both runs selected
                        run_a_id = run_a.get("slurm_job_id", "Unknown")
                        run_b_id = run_b.get("slurm_job_id", "Unknown")
                        selected_for_comparison = [run_a_id, run_b_id]
                        comparison_df = df[df["Run ID"].isin(selected_for_comparison)]

                        if not comparison_df.empty:
                            # Pareto comparison
                            st.markdown("**Pareto Frontier**")
                            pareto_comparison_fig = create_pareto_graph(
                                comparison_df,
                                selected_for_comparison,
                                show_cutoff=False,
                                cutoff_value=30.0,
                                show_frontier=False,
                            )
                            st.plotly_chart(
                                pareto_comparison_fig, width="stretch", key="comparison_pareto"
                            )

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
                                st.plotly_chart(
                                    fig, width="stretch", key=f"comparison_latency_{metric_name}"
                                )

                    with viz_tab2:
                        st.markdown("#### Delta Analysis (Run B - Run A)")
                        st.caption(
                            "Positive values indicate Run B is higher. For latency, negative is better. For throughput, positive is better."
                        )

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
                                fig_ttft.add_hline(
                                    y=0,
                                    line_dash="dash",
                                    line_color="gray",
                                    annotation_text="No change",
                                )
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
                                fig_tpot.add_hline(
                                    y=0,
                                    line_dash="dash",
                                    line_color="gray",
                                    annotation_text="No change",
                                )
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
                                fig_tps.add_hline(
                                    y=0,
                                    line_dash="dash",
                                    line_color="gray",
                                    annotation_text="No change",
                                )
                                fig_tps.update_layout(
                                    title="Output Throughput Delta (Run B - Run A)",
                                    xaxis_title="Concurrency",
                                    yaxis_title="Delta Throughput (tokens/s)",
                                    height=400,
                                )
                                st.plotly_chart(fig_tps, width="stretch", key="delta_throughput")
                        else:
                            st.info("No matching concurrency levels between the two runs.")
                else:
                    st.warning(
                        "No matching concurrency levels found between the two runs to compare."
                    )


if __name__ == "__main__":
    main()
