# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for worker setup."""

import logging
import subprocess

import requests


def get_wheel_arch_from_gpu_type(gpu_type: str) -> str:
    """
    Map GPU type to the appropriate wheel architecture suffix.

    Args:
        gpu_type: GPU type string (e.g., "gb200-fp8", "gb300-fp8", "h100-fp8")

    Returns:
        "aarch64" for GB200/GB300, "x86_64" for H100
    """
    if gpu_type.startswith("gb200") or gpu_type.startswith("gb300"):
        return "aarch64"
    elif gpu_type.startswith("h100"):
        return "x86_64"
    else:
        raise RuntimeError(f"Unknown GPU type: {gpu_type}. Cannot determine wheel architecture.")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s| %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def check_etcd_health(etcd_url: str) -> bool:
    """Check if etcd is healthy"""
    health_url = f"{etcd_url}/health"
    try:
        response = requests.get(health_url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def wait_for_etcd(etcd_url: str, max_retries: int = 1000) -> bool:
    """Wait for etcd to be ready"""
    logging.info(f"Waiting for etcd to be ready on {etcd_url}...")

    for attempt in range(max_retries):
        try:
            if check_etcd_health(etcd_url):
                logging.info("Etcd is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        logging.info(f"Etcd not ready yet, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
        import time

        time.sleep(2)

    return False


def run_command(
    command: str,
    background: bool = False,
    stdout=None,
    stderr=None,
) -> subprocess.Popen | int:
    """
    Run a command in a subprocess.

    Args:
        command: The command to run
        background: If True, run in background and return Popen object
        stdout: Optional stdout file handle
        stderr: Optional stderr file handle

    Returns:
        If background=True: Popen object
        If background=False: Return code
    """
    logging.info(f"Running command: {command}")

    if background:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=stdout if stdout else subprocess.DEVNULL,
            stderr=stderr if stderr else subprocess.DEVNULL,
        )
        return process
    else:
        result = subprocess.run(command, shell=True)
        return result.returncode
