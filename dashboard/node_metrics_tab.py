"""
Node-Level Metrics Tab
"""

import os
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from dashboard.components import (
    load_node_metrics,
    create_node_throughput_graph,
    create_kv_cache_utilization_graph,
    create_queue_depth_graph,
    create_node_inflight_requests_graph,
    create_decode_running_requests_graph,
    create_decode_gen_throughput_graph,
    create_decode_transfer_req_graph,
    create_decode_prealloc_req_graph,
    create_decode_disagg_stacked_graph,
)
from srtslurm.visualizations import aggregate_all_nodes, group_nodes_by_dp


def render(filtered_runs: list, logs_dir: str):
    """Render the node-level metrics tab.
    
    Args:
        filtered_runs: List of BenchmarkRun objects
        logs_dir: Path to logs directory
    """
    st.subheader("Node-Level Metrics")
    st.markdown("""
    Runtime metrics extracted from log files, split by prefill and decode nodes.
    """)
    
    # Parse log files for all selected runs (cached)
    all_node_metrics = []
    with st.spinner(f"Parsing logs for {len(filtered_runs)} run(s)..."):
        for run in filtered_runs:
            run_path = run.metadata.path
            run_id = f"{run.job_id}_{run.metadata.prefill_workers}P_{run.metadata.decode_workers}D_{run.metadata.run_date}"
            if run_path and os.path.exists(run_path):
                node_metrics = load_node_metrics(run_path)
                for node_data in node_metrics:
                    node_data["run_id"] = run_id
                all_node_metrics.extend(node_metrics)
    
    if not all_node_metrics:
        st.warning("No log files (.err) found for the selected runs.")
        st.info("Node metrics are extracted from files like `*_prefill_*.err` and `*_decode_*.err`")
        return
    
    # Split by prefill vs decode
    prefill_nodes = [n for n in all_node_metrics if n["node_info"]["worker_type"] == "prefill"]
    decode_nodes = [n for n in all_node_metrics if n["node_info"]["worker_type"] == "decode"]
    
    st.caption(f"📊 Found {len(prefill_nodes)} prefill nodes, {len(decode_nodes)} decode nodes")
    
    # Add toggle for aggregation
    aggregation_mode = st.radio(
        "📊 Node Aggregation",
        options=[
            "Show individual nodes",
            "Group by DP rank (average per DP)",
            "Aggregate all nodes (single averaged line)",
        ],
        index=2,
        horizontal=True,
        help="Control how node metrics are displayed: individual lines, grouped by DP rank, or fully aggregated across all nodes.",
    )
    
    # Pre-aggregate nodes ONCE to avoid recomputing for each graph
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
        _render_prefill_metrics(prefill_nodes, group_by_dp, aggregate_all)
    
    # Decode Metrics Section
    if decode_nodes:
        st.divider()
        st.markdown("### 📥 Decode Node Metrics")
        _render_decode_metrics(decode_nodes, prefill_nodes, group_by_dp, aggregate_all)


def _render_prefill_metrics(prefill_nodes, group_by_dp, aggregate_all):
    """Render prefill node metrics."""
    throughput_fig = create_node_throughput_graph(prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    throughput_fig.update_xaxes(showgrid=True)
    throughput_fig.update_yaxes(showgrid=True)
    st.plotly_chart(throughput_fig, width="stretch", key="prefill_throughput")
    st.caption("Shows prefill throughput in tokens/s - measures how fast the system processes input prompts")
    
    inflight_fig = create_node_inflight_requests_graph(prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    inflight_fig.update_xaxes(showgrid=True)
    inflight_fig.update_yaxes(showgrid=True)
    st.plotly_chart(inflight_fig, width="stretch", key="prefill_inflight")
    st.caption("Requests that have been sent to decode workers in PD disaggregation mode")
    
    kv_fig = create_kv_cache_utilization_graph(prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    kv_fig.update_xaxes(showgrid=True)
    kv_fig.update_yaxes(showgrid=True)
    st.plotly_chart(kv_fig, width="stretch", key="prefill_kv_util")
    st.caption("Percentage of KV cache memory currently in use - helps tune max-total-tokens and identify memory pressure")
    
    queue_fig = create_queue_depth_graph(prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    queue_fig.update_layout(title="PREFILL Queued Requests")
    queue_fig.update_xaxes(showgrid=True)
    queue_fig.update_yaxes(showgrid=True)
    st.plotly_chart(queue_fig, width="stretch", key="prefill_queue_v2")
    st.caption("Prefill requests waiting in queue - growing queue indicates backpressure or overload")


def _render_decode_metrics(decode_nodes, prefill_nodes, group_by_dp, aggregate_all):
    """Render decode node metrics."""
    # Check if decode nodes have batch data
    has_data = any(node_data["prefill_batches"] for node_data in decode_nodes)
    if not has_data:
        st.warning("⚠️ No batch metrics found for decode nodes. Decode nodes may not log batch-level metrics in the current setup.")
        return
    
    running_fig = create_decode_running_requests_graph(decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    running_fig.update_xaxes(showgrid=True)
    running_fig.update_yaxes(showgrid=True)
    st.plotly_chart(running_fig, width="stretch", key="decode_running")
    st.caption("Number of requests currently being decoded and generating output tokens")
    
    gen_fig = create_decode_gen_throughput_graph(decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    gen_fig.update_xaxes(showgrid=True)
    gen_fig.update_yaxes(showgrid=True)
    st.plotly_chart(gen_fig, width="stretch", key="decode_gen_throughput")
    st.caption("Output token generation rate in tokens/s - measures decode performance")
    
    kv_decode_fig = create_kv_cache_utilization_graph(decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    kv_decode_fig.update_xaxes(showgrid=True)
    kv_decode_fig.update_yaxes(showgrid=True)
    st.plotly_chart(kv_decode_fig, width="stretch", key="decode_kv_util")
    st.caption("Total KV cache tokens in use across all running requests - low indicates underutilization, high indicates risk of OOM")
    
    queue_decode_fig = create_queue_depth_graph(decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    queue_decode_fig.update_layout(title="DECODE Queued Requests")
    queue_decode_fig.update_xaxes(showgrid=True)
    queue_decode_fig.update_yaxes(showgrid=True)
    st.plotly_chart(queue_decode_fig, width="stretch", key="decode_queue_v2")
    st.caption("Decode requests waiting in queue - indicates decode capacity constraints")
    
    # Rate Matching Graph
    st.divider()
    st.markdown("#### Rate Match")
    st.caption("Compare prefill input rate vs decode generation rate to verify proper node ratio")
    
    rate_fig = _create_rate_match_graph(prefill_nodes, decode_nodes)
    st.plotly_chart(rate_fig, width="stretch", key="rate_match")
    st.caption(f"Rate matched when lines align. Prefill: {len(prefill_nodes)} node(s), Decode: {len(decode_nodes)} node(s)")
    
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
        stacked_fig = create_decode_disagg_stacked_graph(decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
        stacked_fig.update_xaxes(showgrid=True)
        stacked_fig.update_yaxes(showgrid=True)
        st.plotly_chart(stacked_fig, width="stretch", key="decode_disagg_stacked")
        st.caption("Shows the request flow funnel: Prealloc Queue → Transfer Queue → Running requests in PD disaggregation")
    else:
        transfer_fig = create_decode_transfer_req_graph(decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
        transfer_fig.update_xaxes(showgrid=True)
        transfer_fig.update_yaxes(showgrid=True)
        st.plotly_chart(transfer_fig, width="stretch", key="decode_transfer")
        st.caption("Requests waiting for KV cache transfer from prefill to decode workers in PD disaggregation mode")
        
        prealloc_fig = create_decode_prealloc_req_graph(decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
        prealloc_fig.update_xaxes(showgrid=True)
        prealloc_fig.update_yaxes(showgrid=True)
        st.plotly_chart(prealloc_fig, width="stretch", key="decode_prealloc")
        st.caption("Requests in pre-allocation queue for PD disaggregation - waiting for memory allocation on decode workers")


def _create_rate_match_graph(prefill_nodes, decode_nodes):
    """Create rate matching graph comparing prefill input vs decode generation."""
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
                first_time = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
                elapsed = [
                    (datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - first_time).total_seconds()
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
    
    # Get decode gen throughput over time
    if decode_nodes:
        all_decode_batches = {}
        for d_node in decode_nodes:
            for batch in d_node["prefill_batches"]:
                if batch.get("gen_throughput") is not None and batch.get("gen_throughput") > 0:
                    ts = batch.get("timestamp", "")
                    if ts:
                        if ts not in all_decode_batches:
                            all_decode_batches[ts] = []
                        all_decode_batches[ts].append(batch["gen_throughput"])
        
        timestamps = []
        total_gen_tps = []
        num_decode = len(decode_nodes)
        
        for ts in sorted(all_decode_batches.keys()):
            avg_gen = sum(all_decode_batches[ts]) / len(all_decode_batches[ts])
            total_gen = avg_gen * num_decode
            timestamps.append(ts)
            total_gen_tps.append(total_gen)
        
        if timestamps:
            first_time = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
            elapsed = [
                (datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - first_time).total_seconds()
                for ts in timestamps
            ]
            
            rate_fig.add_trace(
                go.Scatter(
                    x=elapsed,
                    y=total_gen_tps,
                    mode="lines+markers",
                    name=f"Decode Gen (tok/s) × {num_decode} nodes",
                    line={"color": "green", "width": 2},
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
    
    return rate_fig
