#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for SGLang command generation from YAML configs.

Tests verify that the generated commands.sh files contain the expected
flags and environment variables for both aggregated and disaggregated modes.
"""

import tempfile
from pathlib import Path

from srtctl.backends.sglang import SGLangBackend
from srtctl.core.config import load_config


def test_basic_disaggregated_commands():
    """Test command generation for basic disaggregated mode (1P + 4D)."""

    # Create minimal disaggregated config
    config = {
        "name": "test-disagg",
        "model": {
            "path": "/models/test-model",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "prefill_nodes": 1,
            "decode_nodes": 4,
            "prefill_workers": 1,
            "decode_workers": 4,
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test-account",
            "partition": "test-partition",
            "time_limit": "01:00:00"
        },
        "backend": {
            "type": "sglang",
            "prefill_environment": {
                "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "1800",
                "PYTHONUNBUFFERED": "1",
                "SGLANG_ENABLE_FLASHINFER_GEMM": "1"
            },
            "decode_environment": {
                "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "1800",
                "PYTHONUNBUFFERED": "1",
                "SGLANG_ENABLE_FLASHINFER_GEMM": "1"
            },
            "sglang_config": {
                "prefill": {
                    "served-model-name": "test-model",
                    "model-path": "/model/",
                    "trust-remote-code": True,
                    "kv-cache-dtype": "fp8_e4m3",
                    "mem-fraction-static": 0.95,
                    "quantization": "fp8",
                    "disaggregation-mode": "prefill",
                    "max-total-tokens": 8192,
                    "chunked-prefill-size": 8192,
                    "tensor-parallel-size": 4,
                    "data-parallel-size": 1
                },
                "decode": {
                    "served-model-name": "test-model",
                    "model-path": "/model/",
                    "trust-remote-code": True,
                    "kv-cache-dtype": "fp8_e4m3",
                    "mem-fraction-static": 0.95,
                    "quantization": "fp8",
                    "disaggregation-mode": "decode",
                    "chunked-prefill-size": 8192,
                    "tensor-parallel-size": 4,
                    "data-parallel-size": 1
                }
            }
        },
        "benchmark": {
            "type": "manual"
        }
    }

    # Create backend and generate config
    backend = SGLangBackend(config)
    sglang_config_path = backend.generate_config_file()

    # Render commands
    prefill_cmd = backend.render_command(mode="prefill", config_path=sglang_config_path)
    decode_cmd = backend.render_command(mode="decode", config_path=sglang_config_path)

    # Verify prefill command
    assert "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800" in prefill_cmd
    assert "PYTHONUNBUFFERED=1" in prefill_cmd
    assert "SGLANG_ENABLE_FLASHINFER_GEMM=1" in prefill_cmd
    assert "python3 -m dynamo.sglang" in prefill_cmd
    assert "--disaggregation-mode prefill" in prefill_cmd
    assert "--tensor-parallel-size 4" in prefill_cmd
    assert "--max-total-tokens 8192" in prefill_cmd
    assert "--chunked-prefill-size 8192" in prefill_cmd
    assert "--mem-fraction-static 0.95" in prefill_cmd
    assert "--kv-cache-dtype fp8_e4m3" in prefill_cmd
    assert "--quantization fp8" in prefill_cmd
    assert "--nnodes 1" in prefill_cmd  # 1 prefill node
    assert "--dist-init-addr $HOST_IP_MACHINE:$PORT" in prefill_cmd
    assert "--node-rank $RANK" in prefill_cmd

    # Verify decode command
    assert "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800" in decode_cmd
    assert "PYTHONUNBUFFERED=1" in decode_cmd
    assert "SGLANG_ENABLE_FLASHINFER_GEMM=1" in decode_cmd
    assert "python3 -m dynamo.sglang" in decode_cmd
    assert "--disaggregation-mode decode" in decode_cmd
    assert "--tensor-parallel-size 4" in decode_cmd
    assert "--chunked-prefill-size 8192" in decode_cmd
    assert "--mem-fraction-static 0.95" in decode_cmd
    assert "--kv-cache-dtype fp8_e4m3" in decode_cmd
    assert "--quantization fp8" in decode_cmd
    assert "--nnodes 4" in decode_cmd  # 4 decode nodes
    assert "--dist-init-addr $HOST_IP_MACHINE:$PORT" in decode_cmd
    assert "--node-rank $RANK" in decode_cmd

    # Verify max-total-tokens is NOT in decode (prefill-only flag)
    assert "--max-total-tokens" not in decode_cmd

    print("✅ Disaggregated mode command generation test passed")


def test_basic_aggregated_commands():
    """Test command generation for basic aggregated mode."""

    # Create minimal aggregated config
    config = {
        "name": "test-agg",
        "model": {
            "path": "/models/test-model",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "agg_nodes": 4,
            "agg_workers": 4,
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test-account",
            "partition": "test-partition",
            "time_limit": "01:00:00"
        },
        "backend": {
            "type": "sglang",
            "aggregated_environment": {
                "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "1800",
                "PYTHONUNBUFFERED": "1",
                "SGLANG_ENABLE_FLASHINFER_GEMM": "1"
            },
            "sglang_config": {
                "aggregated": {
                    "served-model-name": "test-model",
                    "model-path": "/model/",
                    "trust-remote-code": True,
                    "kv-cache-dtype": "fp8_e4m3",
                    "mem-fraction-static": 0.95,
                    "quantization": "fp8",
                    "max-total-tokens": 16384,
                    "chunked-prefill-size": 8192,
                    "tensor-parallel-size": 4,
                    "data-parallel-size": 1
                }
            }
        },
        "benchmark": {
            "type": "manual"
        }
    }

    # Create backend and generate config
    backend = SGLangBackend(config)
    sglang_config_path = backend.generate_config_file()

    # Render aggregated command
    agg_cmd = backend.render_command(mode="aggregated", config_path=sglang_config_path)

    # Verify aggregated command
    assert "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800" in agg_cmd
    assert "PYTHONUNBUFFERED=1" in agg_cmd
    assert "SGLANG_ENABLE_FLASHINFER_GEMM=1" in agg_cmd
    assert "python3 -m dynamo.sglang" in agg_cmd
    assert "--tensor-parallel-size 4" in agg_cmd
    assert "--data-parallel-size 1" in agg_cmd
    assert "--max-total-tokens 16384" in agg_cmd
    assert "--chunked-prefill-size 8192" in agg_cmd
    assert "--mem-fraction-static 0.95" in agg_cmd
    assert "--kv-cache-dtype fp8_e4m3" in agg_cmd
    assert "--quantization fp8" in agg_cmd
    assert "--nnodes 4" in agg_cmd  # 4 aggregated nodes
    assert "--dist-init-addr $HOST_IP_MACHINE:$PORT" in agg_cmd
    assert "--node-rank $RANK" in agg_cmd

    # Verify disaggregation-mode is NOT present in aggregated
    assert "--disaggregation-mode" not in agg_cmd

    print("✅ Aggregated mode command generation test passed")


def test_environment_variable_handling():
    """Test that environment variables are correctly handled."""

    # Config with no environment variables
    config = {
        "name": "test-no-env",
        "model": {
            "path": "/models/test",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "prefill_nodes": 1,
            "decode_nodes": 1,
            "prefill_workers": 1,
            "decode_workers": 1,
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test",
            "partition": "test",
            "time_limit": "01:00:00"
        },
        "backend": {
            "type": "sglang",
            "sglang_config": {
                "prefill": {
                    "model-path": "/model/",
                    "tensor-parallel-size": 4
                }
            }
        },
        "benchmark": {"type": "manual"}
    }

    backend = SGLangBackend(config)
    sglang_config_path = backend.generate_config_file()
    cmd = backend.render_command(mode="prefill", config_path=sglang_config_path)

    # Should have python command without env vars
    assert "python3 -m dynamo.sglang" in cmd
    # Should not have any env var lines before python
    lines = cmd.split("\n")
    python_line_idx = None
    for i, line in enumerate(lines):
        if "python3" in line:
            python_line_idx = i
            break

    # All lines before python should not contain "="
    if python_line_idx:
        for i in range(python_line_idx):
            line = lines[i].strip()
            if line:  # Non-empty line
                assert "=" not in line, f"Found env var before python: {line}"

    print("✅ Environment variable handling test passed")


def test_profiling_mode():
    """Test that profiling mode uses sglang.launch_server."""

    config = {
        "name": "test-profiling",
        "model": {
            "path": "/models/test",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "prefill_nodes": 1,
            "decode_nodes": 1,
            "prefill_workers": 1,
            "decode_workers": 1,
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test",
            "partition": "test",
            "time_limit": "01:00:00"
        },
        "backend": {
            "type": "sglang",
            "enable_profiling": True,  # Enable profiling
            "sglang_config": {
                "prefill": {
                    "model-path": "/model/",
                    "tensor-parallel-size": 4,
                    "disaggregation-mode": "prefill"
                }
            }
        },
        "benchmark": {"type": "manual"}
    }

    backend = SGLangBackend(config)
    sglang_config_path = backend.generate_config_file()
    cmd = backend.render_command(mode="prefill", config_path=sglang_config_path)

    # Should use sglang.launch_server instead of dynamo.sglang
    assert "python3 -m sglang.launch_server" in cmd
    assert "python3 -m dynamo.sglang" not in cmd

    # Should skip disaggregation-mode flag when profiling
    assert "--disaggregation-mode" not in cmd

    print("✅ Profiling mode test passed")


def test_config_from_yaml_file():
    """Test loading config from actual YAML file and generating commands."""

    # Use the example.yaml config
    config_path = Path(__file__).parent.parent / "configs" / "example.yaml"

    if not config_path.exists():
        print("⚠️  Skipping test_config_from_yaml_file - example.yaml not found")
        return

    config = load_config(config_path)

    # Create backend and generate commands
    backend = SGLangBackend(config)
    sglang_config_path = backend.generate_config_file()

    prefill_cmd = backend.render_command(mode="prefill", config_path=sglang_config_path)
    decode_cmd = backend.render_command(mode="decode", config_path=sglang_config_path)

    # Basic sanity checks
    assert "python3 -m dynamo.sglang" in prefill_cmd
    assert "python3 -m dynamo.sglang" in decode_cmd
    assert "--disaggregation-mode prefill" in prefill_cmd
    assert "--disaggregation-mode decode" in decode_cmd
    assert "--nnodes 1" in prefill_cmd
    assert "--nnodes 4" in decode_cmd

    print("✅ YAML file loading and command generation test passed")


if __name__ == "__main__":
    # Run tests when executed directly
    print("Running command generation tests...\n")

    test_basic_disaggregated_commands()
    test_basic_aggregated_commands()
    test_environment_variable_handling()
    test_profiling_mode()
    test_config_from_yaml_file()

    print("\n✅ All command generation tests passed!")
