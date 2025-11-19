"""
Test aggregation and grouping functionality
"""

import os

from srtlog import NodeAnalyzer
from srtlog.visualizations import aggregate_all_nodes, group_nodes_by_dp


def test_group_by_dp():
    """Test grouping nodes by DP rank."""
    test_dir = os.path.dirname(__file__)
    run_dir = os.path.join(test_dir, "9999_1P_2D_20251110_120000")

    analyzer = NodeAnalyzer()
    nodes_obj = analyzer.parse_run_logs(run_dir)

    # Convert to dicts for visualization functions
    nodes: list[dict] = [
        {
            "node_info": n.node_info,
            "prefill_batches": [
                {
                    "timestamp": b.timestamp,
                    "dp": b.dp,
                    "tp": b.tp,
                    "ep": b.ep,
                    "type": b.batch_type,
                    "running_req": b.running_req,
                    "gen_throughput": b.gen_throughput,
                }
                for b in n.batches
            ],
            "memory_snapshots": [],
            "config": n.config,
            "run_id": "9999",
        }
        for n in nodes_obj
    ]

    decode_nodes = [n for n in nodes if n["node_info"]["worker_type"] == "decode"]

    # Group by DP - should combine TP0, TP1, TP2 within each DP
    grouped = group_nodes_by_dp(decode_nodes)

    # Should have 2 groups (DP0 and DP1)
    assert len(grouped) == 2, f"Expected 2 DP groups, got {len(grouped)}"

    # Find DP0 and DP1 groups
    dp0_group = [g for g in grouped if g["prefill_batches"][0]["dp"] == 0][0]
    dp1_group = [g for g in grouped if g["prefill_batches"][0]["dp"] == 1][0]

    # Each group should have averaged values at matching timestamps
    # DP0 has 3 batches at different times (one per TP rank)
    assert len(dp0_group["prefill_batches"]) == 3

    # DP1 has 3 batches at different times (one per TP rank)
    assert len(dp1_group["prefill_batches"]) == 3


def test_aggregate_all():
    """Test aggregating all nodes together."""
    test_dir = os.path.dirname(__file__)
    run_dir = os.path.join(test_dir, "9999_1P_2D_20251110_120000")

    analyzer = NodeAnalyzer()
    nodes_obj = analyzer.parse_run_logs(run_dir)

    # Convert to dicts for visualization functions
    nodes: list[dict] = [
        {
            "node_info": n.node_info,
            "prefill_batches": [
                {
                    "timestamp": b.timestamp,
                    "dp": b.dp,
                    "tp": b.tp,
                    "ep": b.ep,
                    "type": b.batch_type,
                    "running_req": b.running_req,
                    "gen_throughput": b.gen_throughput,
                }
                for b in n.batches
            ],
            "memory_snapshots": [],
            "config": n.config,
            "run_id": "9999",
        }
        for n in nodes_obj
    ]

    decode_nodes = [n for n in nodes if n["node_info"]["worker_type"] == "decode"]

    # Aggregate all - should combine everything into one averaged line
    aggregated = aggregate_all_nodes(decode_nodes)

    # Should have 1 aggregated node (all DP and TP ranks combined)
    assert len(aggregated) == 1, f"Expected 1 aggregated node, got {len(aggregated)}"

    agg = aggregated[0]
    assert "ALL_NODES" in agg["node_info"]["node"]
    assert "avg_2_nodes" in agg["node_info"]["worker_id"]

    # Should have batches (averaged across all nodes)
    assert len(agg["prefill_batches"]) > 0


def test_batch_metrics_calculations():
    """Test batch metric calculations."""
    test_dir = os.path.dirname(__file__)
    run_dir = os.path.join(test_dir, "9999_1P_2D_20251110_120000")

    analyzer = NodeAnalyzer()
    nodes = analyzer.parse_run_logs(run_dir)

    prefill = analyzer.get_prefill_nodes(nodes)
    p = prefill[0]

    # Find batch with cached tokens
    batch_with_cache = [b for b in p.batches if b.cached_token and b.cached_token > 0][0]

    # Verify cache hit rate calculation
    assert batch_with_cache.new_token is not None
    assert batch_with_cache.cached_token is not None
    expected_rate = (
        batch_with_cache.cached_token / (batch_with_cache.new_token + batch_with_cache.cached_token)
    ) * 100
    assert batch_with_cache.cache_hit_rate == expected_rate


if __name__ == "__main__":
    # Run tests when executed directly
    test_group_by_dp()
    print("✅ test_group_by_dp passed")

    test_aggregate_all()
    print("✅ test_aggregate_all passed")

    test_batch_metrics_calculations()
    print("✅ test_batch_metrics_calculations passed")

    print("\n✅ All aggregation tests passed!")
