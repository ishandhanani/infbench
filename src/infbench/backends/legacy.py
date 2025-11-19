#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Legacy conversion functions for backward compatibility with submit_job_script.py.

TODO: Remove once migration is complete.
"""

import json
from pathlib import Path


def yaml_to_args(config: dict, sglang_config_path: Path = None) -> list[str]:
    """Convert YAML config to submit_job_script.py arguments.
    
    LEGACY: This function exists for backward compatibility only.
    """
    args = [
        "--job-name", config['name'],
        "--account", config['slurm']['account'],
        "--partition", config['slurm']['partition'],
        "--time-limit", config['slurm']['time_limit'],
        "--model-dir", config['model']['path'],
        "--container-image", config['model']['container'],
        "--gpus-per-node", str(config['resources']['gpus_per_node']),
    ]

    # Mode: aggregated or disaggregated
    if 'agg_nodes' in config['resources']:
        args.extend([
            "--agg-nodes", str(config['resources']['agg_nodes']),
            "--agg-workers", str(config['resources']['agg_workers']),
        ])
    else:
        args.extend([
            "--prefill-nodes", str(config['resources']['prefill_nodes']),
            "--decode-nodes", str(config['resources']['decode_nodes']),
            "--prefill-workers", str(config['resources']['prefill_workers']),
            "--decode-workers", str(config['resources']['decode_workers']),
        ])

    backend = config.get('backend', {})
    if 'gpu_type' in backend:
        args.extend(["--gpu-type", backend['gpu_type']])
    if 'script_variant' in backend:
        args.extend(["--script-variant", backend['script_variant']])

    if sglang_config_path:
        args.extend(["--sglang-config-path", str(sglang_config_path)])

    if 'environment' in backend:
        env_json = json.dumps(backend['environment'])
        args.extend(["--backend-env", env_json])

    if 'benchmark' in config:
        benchmark_str = format_benchmark(config['benchmark'])
        args.extend(["--benchmark", benchmark_str])

    if config.get('use_init_location', False):
        args.append("--use-init-location")

    if not config.get('enable_config_dump', True):
        args.append("--disable-config-dump")

    return args


def format_benchmark(bench: dict) -> str:
    """Format benchmark dict to string.
    
    LEGACY: This function exists for backward compatibility only.
    """
    bench_type = bench.get('type', 'manual')

    if bench_type == 'sa-bench':
        concurrencies = bench['concurrencies']
        if isinstance(concurrencies, list):
            concurrency_str = "x".join(str(c) for c in concurrencies)
        else:
            concurrency_str = str(concurrencies)

        return (
            f"type=sa-bench; "
            f"isl={bench['isl']}; "
            f"osl={bench['osl']}; "
            f"concurrencies={concurrency_str}; "
            f"req-rate={bench['req_rate']}"
        )
    elif bench_type == 'manual':
        return "type=manual"
    else:
        raise ValueError(f"Unknown benchmark type: {bench_type}")
