"""
Basic tests for class-based architecture
"""

import os

from srtlog import NodeAnalyzer, RunLoader


def test_run_loader():
    """Test RunLoader can load test fixture."""
    # Use tests directory
    test_dir = os.path.dirname(__file__)

    loader = RunLoader(test_dir)

    # Test loading all runs
    runs = loader.load_all()
    assert len(runs) == 2, f"Expected 2 runs, got {len(runs)}"

    # Test run 9999 metadata
    run_9999 = [r for r in runs if r.job_id == "9999"][0]
    assert run_9999.job_id == "9999"
    assert run_9999.metadata.prefill_nodes == 1
    assert run_9999.metadata.decode_nodes == 2
    assert run_9999.metadata.gpus_per_node == 4
    assert run_9999.metadata.total_gpus == 12  # (1 + 2) * 4
    assert run_9999.metadata.formatted_date == "Nov 10"

    # Test profiler results
    assert run_9999.profiler.profiler_type == "vllm"
    assert run_9999.profiler.isl == "512"
    assert run_9999.profiler.osl == "256"
    assert len(run_9999.profiler.output_tps) == 2
    assert run_9999.profiler.output_tps[0] == 1234.5
    assert run_9999.profiler.output_tps[1] == 2345.6


def test_node_analyzer():
    """Test NodeAnalyzer can parse test fixture logs."""
    test_dir = os.path.dirname(__file__)
    run_dir = os.path.join(test_dir, "9999_1P_2D_20251110_120000")

    analyzer = NodeAnalyzer()

    # Test parsing logs
    nodes = analyzer.parse_run_logs(run_dir)
    assert len(nodes) == 3, f"Expected 3 nodes, got {len(nodes)}"

    # Test filtering
    prefill = analyzer.get_prefill_nodes(nodes)
    decode = analyzer.get_decode_nodes(nodes)

    assert len(prefill) == 1
    assert len(decode) == 2

    # Test prefill node
    p = prefill[0]
    assert p.is_prefill is True
    assert p.is_decode is False
    assert p.worker_type == "prefill"

    # Test batch metrics - should have 3 batches with different TP values
    assert len(p.batches) == 3
    batch = p.batches[0]
    assert batch.batch_type == "prefill"
    assert batch.dp == 0
    assert batch.tp == 0
    assert batch.new_seq == 10
    assert batch.new_token == 5120
    assert batch.input_throughput == 1024.5

    # Test different TP ranks
    batch2 = p.batches[1]
    assert batch2.tp == 1  # Different TP rank
    assert batch2.cache_hit_rate == 20.0  # 1024 / (4096 + 1024) * 100

    batch3 = p.batches[2]
    assert batch3.tp == 2  # Yet another TP rank

    # Test decode nodes - should have different DP ranks
    assert len(decode) == 2
    d0 = [d for d in decode if d.batches and d.batches[0].dp == 0][0]
    d1 = [d for d in decode if d.batches and d.batches[0].dp == 1][0]

    # Decode node with DP0
    assert len(d0.batches) == 3  # 3 TP ranks
    assert d0.batches[0].dp == 0
    assert d0.batches[0].tp == 0
    assert d0.batches[1].tp == 1
    assert d0.batches[2].tp == 2

    # Decode node with DP1
    assert len(d1.batches) == 3  # 3 TP ranks
    assert d1.batches[0].dp == 1
    assert d1.batches[0].tp == 0
    assert d1.batches[1].tp == 1
    assert d1.batches[2].tp == 2


def test_node_count():
    """Test node counting functionality."""
    test_dir = os.path.dirname(__file__)
    run_dir = os.path.join(test_dir, "9999_1P_2D_20251110_120000")

    analyzer = NodeAnalyzer()
    prefill_count, decode_count = analyzer.get_node_count(run_dir)

    assert prefill_count == 1
    assert decode_count == 2


def test_incomplete_job_detection():
    """Test that incomplete jobs are detected correctly."""
    test_dir = os.path.dirname(__file__)

    loader = RunLoader(test_dir)
    runs = loader.load_all()
    assert len(runs) == 2, f"Expected 2 runs, got {len(runs)}"

    # Sort by job_id to get consistent ordering
    runs = sorted(runs, key=lambda r: r.job_id)

    # Test run 8888 - INCOMPLETE (expects 128x256x512, only has 128 and 256)
    run_8888 = [r for r in runs if r.job_id == "8888"][0]
    assert run_8888.profiler.concurrencies == "128x256x512"
    assert len(run_8888.profiler.concurrency_values) == 2
    assert set(run_8888.profiler.concurrency_values) == {128, 256}

    # Should be marked as INCOMPLETE with missing concurrency 512
    assert run_8888.is_complete is False
    assert run_8888.missing_concurrencies == [512]

    # Test run 9999 - COMPLETE (expects 128x256, has both)
    run_9999 = [r for r in runs if r.job_id == "9999"][0]
    assert run_9999.profiler.concurrencies == "128x256"
    assert len(run_9999.profiler.concurrency_values) == 2
    assert set(run_9999.profiler.concurrency_values) == {128, 256}

    # Should be marked as complete
    assert run_9999.is_complete is True
    assert run_9999.missing_concurrencies == []


if __name__ == "__main__":
    # Run tests when executed directly
    test_run_loader()
    print("✅ test_run_loader passed")

    test_node_analyzer()
    print("✅ test_node_analyzer passed")

    test_node_count()
    print("✅ test_node_count passed")

    test_incomplete_job_detection()
    print("✅ test_incomplete_job_detection passed")

    print("\n✅ All basic tests passed!")
