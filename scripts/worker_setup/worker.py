# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker setup functions for prefill, decode, and aggregated workers."""

import logging

from .command import get_gpu_command, install_dynamo_wheels
from .environment import DIST_INIT_PORT, ETCD_CLIENT_PORT
from .infrastructure import setup_head_prefill_node
from .utils import run_command, wait_for_etcd


def setup_prefill_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpu_type: str,
    multiple_frontends_enabled: bool = False,
    sglang_torch_profiler: bool = False,
    sglang_config_path: str | None = None,
) -> int:
    """Setup the prefill worker."""
    # Only setup infrastructure in traditional mode (not multiple frontends)
    if not multiple_frontends_enabled and worker_idx == 0 and local_rank == 0:
        setup_head_prefill_node(master_ip)
    else:
        logging.info(f"Setting up prefill worker {worker_idx}, local rank {local_rank}")
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # Install dynamo wheels
    install_dynamo_wheels(gpu_type)

    # Build and execute SGLang command from YAML config
    cmd_to_run = get_gpu_command(
        worker_type="prefill",
        sglang_config_path=sglang_config_path,
        host_ip=leader_ip,
        port=DIST_INIT_PORT,
        total_nodes=nodes_per_worker,
        rank=local_rank,
        use_profiling=sglang_torch_profiler,
    )
    return run_command(cmd_to_run)


def setup_decode_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpu_type: str,
    sglang_torch_profiler: bool = False,
    sglang_config_path: str | None = None,
) -> int:
    """Setup the decode worker."""
    logging.info(f"Setting up decode worker {worker_idx}, local rank {local_rank}")

    if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
        raise RuntimeError("Failed to connect to etcd")

    # Install dynamo wheels
    install_dynamo_wheels(gpu_type)

    # Build and execute SGLang command from YAML config
    cmd_to_run = get_gpu_command(
        worker_type="decode",
        sglang_config_path=sglang_config_path,
        host_ip=leader_ip,
        port=DIST_INIT_PORT,
        total_nodes=nodes_per_worker,
        rank=local_rank,
        use_profiling=sglang_torch_profiler,
    )
    return run_command(cmd_to_run)


def setup_aggregated_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpu_type: str,
    multiple_frontends_enabled: bool = False,
    sglang_torch_profiler: bool = False,
    sglang_config_path: str | None = None,
) -> int:
    """Setup the aggregated worker."""
    # Only setup infrastructure in traditional mode (not multiple frontends) on first worker, first node
    if not multiple_frontends_enabled and worker_idx == 0 and local_rank == 0:
        setup_head_prefill_node(master_ip)
    else:
        logging.info(f"Setting up aggregated worker {worker_idx}, local rank {local_rank}")
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # Install dynamo wheels
    install_dynamo_wheels(gpu_type)

    # Build and execute SGLang command from YAML config
    cmd_to_run = get_gpu_command(
        worker_type="aggregated",
        sglang_config_path=sglang_config_path,
        host_ip=leader_ip,
        port=DIST_INIT_PORT,
        total_nodes=nodes_per_worker,
        rank=local_rank,
        use_profiling=sglang_torch_profiler,
    )
    return run_command(cmd_to_run)
