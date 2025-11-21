# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command building functions for SGLang workers."""

import logging
import os
import subprocess

from .utils import get_wheel_arch_from_gpu_type


def build_sglang_command_from_yaml(
    worker_type: str,
    sglang_config_path: str,
    host_ip: str,
    port: int,
    total_nodes: int,
    rank: int,
    use_profiling: bool = False,
    dump_config_path: str | None = None,
) -> str:
    """Build SGLang command using native YAML config support.

    dynamo.sglang supports reading config from YAML:
        python3 -m dynamo.sglang --config file.yaml --config-key prefill

    sglang.launch_server (profiling mode) requires explicit flags:
        python3 -m sglang.launch_server --model-path /model/ --tp 4 ...

    Args:
        worker_type: "prefill", "decode", or "aggregated"
        sglang_config_path: Path to generated sglang_config.yaml
        host_ip: Host IP for distributed coordination
        port: Port for distributed coordination
        total_nodes: Total number of nodes
        rank: Node rank (0-indexed)
        use_profiling: Whether to use sglang.launch_server (profiling mode)

    Returns:
        Full command string ready to execute
    """
    import yaml

    # Load config to extract environment variables and mode config
    with open(sglang_config_path) as f:
        sglang_config = yaml.safe_load(f)

    config_key = worker_type if worker_type != "aggregated" else "aggregated"

    # Environment variables are stored at top level as {mode}_environment
    env_key = f"{config_key}_environment"
    env_vars = sglang_config.get(env_key, {})

    # Build environment variable exports
    env_exports = []
    for key, value in env_vars.items():
        env_exports.append(f"export {key}={value}")

    # Determine Python module based on profiling mode
    python_module = "sglang.launch_server" if use_profiling else "dynamo.sglang"

    if use_profiling:
        # Profiling mode: inline all flags (sglang.launch_server doesn't support --config)
        mode_config = sglang_config.get(config_key, {})
        cmd_parts = [f"python3 -m {python_module}"]
        
        # Add all SGLang flags from config
        for key, value in sorted(mode_config.items()):
            flag_name = key.replace("_", "-")
            # Skip disaggregation-mode flag for profiling
            if flag_name == "disaggregation-mode":
                continue
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{flag_name}")
            elif isinstance(value, list):
                values_str = " ".join(str(v) for v in value)
                cmd_parts.append(f"--{flag_name} {values_str}")
            else:
                cmd_parts.append(f"--{flag_name} {value}")
        
        # Add coordination flags
        cmd_parts.extend([
            f"--dist-init-addr {host_ip}:{port}",
            f"--nnodes {total_nodes}",
            f"--node-rank {rank}",
            "--host 0.0.0.0",
        ])
    else:
        # Normal mode: use --config and --config-key (dynamo.sglang supports this)
        cmd_parts = [
            f"python3 -m {python_module}",
            f"--config {sglang_config_path}",
            f"--config-key {config_key}",
            f"--dist-init-addr {host_ip}:{port}",
            f"--nnodes {total_nodes}",
            f"--node-rank {rank}",
            "--host 0.0.0.0",
        ]

    # Add dump-config-to flag if provided
    if dump_config_path:
        cmd_parts.append(f"--dump-config-to {dump_config_path}")

    # Combine environment exports and command
    full_command = " && ".join(env_exports + [" ".join(cmd_parts)]) if env_exports else " ".join(cmd_parts)

    return full_command


def install_dynamo_wheels(gpu_type: str) -> None:
    """Install dynamo wheels.

    Args:
        gpu_type: GPU type to determine architecture (e.g., "gb200-fp8", "h100-fp8")
    """
    arch = get_wheel_arch_from_gpu_type(gpu_type)
    logging.info(f"Installing dynamo wheels for architecture: {arch}")

    # Install runtime wheel
    runtime_whl = f"/configs/ai_dynamo_runtime-0.7.0-cp310-abi3-manylinux_2_28_{arch}.whl"
    logging.info(f"Installing {runtime_whl}")
    result = subprocess.run(["python3", "-m", "pip", "install", runtime_whl], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Failed to install runtime wheel: {result.stderr}")
        raise RuntimeError(f"Failed to install {runtime_whl}")

    # Install dynamo wheel
    dynamo_whl = "/configs/ai_dynamo-0.7.0-py3-none-any.whl"
    logging.info(f"Installing {dynamo_whl}")
    result = subprocess.run(["python3", "-m", "pip", "install", dynamo_whl], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Failed to install dynamo wheel: {result.stderr}")
        raise RuntimeError(f"Failed to install {dynamo_whl}")

    logging.info("Successfully installed dynamo wheels")


def get_gpu_command(
    worker_type: str,
    sglang_config_path: str,
    host_ip: str,
    port: int,
    total_nodes: int,
    rank: int,
    use_profiling: bool = False,
    dump_config_path: str | None = None,
) -> str:
    """Generate command to run SGLang worker using YAML config.

    Args:
        worker_type: "prefill", "decode", or "aggregated"
        sglang_config_path: Path to sglang_config.yaml
        host_ip: Host IP for distributed coordination
        port: Port for distributed coordination
        total_nodes: Total number of nodes
        rank: Node rank (0-indexed)
        use_profiling: Whether to use sglang.launch_server (profiling mode)

    Returns:
        Command string to execute
    """
    if not sglang_config_path or not os.path.exists(sglang_config_path):
        raise ValueError(f"SGLang config path required but not found: {sglang_config_path}")

    logging.info(f"Building command from YAML config: {sglang_config_path}")
    return build_sglang_command_from_yaml(
        worker_type, sglang_config_path, host_ip, port, total_nodes, rank, use_profiling, dump_config_path
    )
