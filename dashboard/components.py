"""
Shared components and utilities for the dashboard
"""

import logging
import os

import streamlit as st

from srtlog import NodeAnalyzer, RunLoader
from srtlog.cloud_sync import create_sync_manager_from_config


# Update default config path
DEFAULT_CONFIG = "srtslurm.yaml"

logger = logging.getLogger(__name__)


# Custom CSS
CUSTOM_CSS = """
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
    /* Widen multiselect to prevent truncation */
    [data-baseweb="select"] {
        min-width: 100% !important;
    }
    [data-baseweb="select"] > div {
        max-width: none !important;
    }
    /* Increase max-width of selected items */
    [data-baseweb="tag"] {
        max-width: 400px !important;
    }
</style>
"""


def apply_custom_css():
    """Apply custom CSS to the dashboard."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def sync_cloud_data(logs_dir):
    """Sync missing runs from cloud storage if configured.

    Returns:
        Tuple of (sync_performed: bool, files_downloaded: int, error_message: str or None)
    """
    try:
        sync_manager = create_sync_manager_from_config(DEFAULT_CONFIG)
        if sync_manager is None:
            # No cloud config, skip sync
            return False, 0, None

        # Sync missing runs (only downloads files that don't exist locally)
        runs_synced, files_downloaded, files_skipped = sync_manager.sync_missing_runs(logs_dir)
        
        # Log sync details
        if files_downloaded > 0:
            logger.info(f"Synced {runs_synced} runs: {files_downloaded} downloaded, {files_skipped} skipped")
        else:
            logger.info(f"All runs up to date ({files_skipped} files already present)")
        
        return True, files_downloaded, None
    except Exception as e:
        logger.error(f"Failed to sync cloud data: {e}")
        return True, 0, str(e)


@st.cache_data
def load_data(logs_dir):
    """Load and cache benchmark data.
    
    Returns:
        Tuple of (runs_with_data, skipped_runs)
        - runs_with_data: List of BenchmarkRun objects with benchmark results
        - skipped_runs: List of tuples (job_id, run_dir, reason) for skipped runs
    """
    loader = RunLoader(logs_dir)
    return loader.load_all_with_skipped()


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


def get_default_logs_dir():
    """Get default logs directory path."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


# Cached graph creation functions for node metrics
@st.cache_data(show_spinner=False)
def create_node_throughput_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create input throughput over time graph."""
    from srtlog.visualizations import create_node_metric_graph
    
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
def create_kv_cache_utilization_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create KV cache utilization visualization."""
    from srtlog.visualizations import create_node_metric_graph
    
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
    from srtlog.visualizations import create_node_metric_graph
    
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
    from srtlog.visualizations import create_node_metric_graph
    
    fig = create_node_metric_graph(
        node_metrics_list,
        title="Inflight Requests Over Time",
        y_label="Number of Requests",
        metric_key="inflight_req",
        mode="lines+markers",
        stackgroup=None,  # Don't stack - show individual lines
        group_by_dp=group_by_dp,
        aggregate_all=aggregate_all,
    )
    fig.update_layout(hovermode="x unified")
    return fig


@st.cache_data(show_spinner=False)
def create_decode_running_requests_graph(node_metrics_list, group_by_dp=False, aggregate_all=False):
    """Create running requests visualization for decode nodes."""
    from srtlog.visualizations import create_node_metric_graph
    
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
    from srtlog.visualizations import create_node_metric_graph
    
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
    from srtlog.visualizations import create_node_metric_graph
    
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
    from srtlog.visualizations import create_node_metric_graph
    
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
    from srtlog.visualizations import create_stacked_metric_graph
    
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

