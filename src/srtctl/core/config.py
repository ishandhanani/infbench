#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Config loading and resolution with srtslurm.yaml integration.
"""

import copy
import logging
import yaml
from pathlib import Path


def load_cluster_config() -> dict | None:
    """
    Load cluster configuration from srtslurm.yaml if it exists.

    Returns None if file doesn't exist (graceful degradation).
    """
    # Look for srtslurm.yaml at project root
    cluster_config_path = Path.cwd() / "srtslurm.yaml"

    if not cluster_config_path.exists():
        logging.debug("No srtslurm.yaml found - using config as-is")
        return None

    try:
        with open(cluster_config_path) as f:
            cluster_config = yaml.safe_load(f)
        logging.debug(f"Loaded cluster config from {cluster_config_path}")
        return cluster_config
    except Exception as e:
        logging.warning(f"Failed to load srtslurm.yaml: {e}")
        return None


def resolve_config_with_defaults(user_config: dict, cluster_config: dict | None) -> dict:
    """
    Resolve user config by applying cluster defaults and aliases.

    Args:
        user_config: User's YAML config
        cluster_config: Cluster defaults from srtslurm.yaml (or None)

    Returns:
        Resolved config with all defaults applied
    """
    # Deep copy to avoid mutating original
    config = copy.deepcopy(user_config)

    if cluster_config is None:
        return config

    # Apply SLURM defaults
    slurm = config.setdefault('slurm', {})
    if 'account' not in slurm and 'default_account' in cluster_config:
        slurm['account'] = cluster_config['default_account']
        logging.debug(f"Applied default account: {slurm['account']}")

    if 'partition' not in slurm and 'default_partition' in cluster_config:
        slurm['partition'] = cluster_config['default_partition']
        logging.debug(f"Applied default partition: {slurm['partition']}")

    if 'time_limit' not in slurm and 'default_time_limit' in cluster_config:
        slurm['time_limit'] = cluster_config['default_time_limit']
        logging.debug(f"Applied default time_limit: {slurm['time_limit']}")

    # Resolve model path alias
    model = config.get('model', {})
    model_path = model.get('path', '')

    if 'model_paths' in cluster_config and model_path in cluster_config['model_paths']:
        resolved_path = cluster_config['model_paths'][model_path]
        model['path'] = resolved_path
        logging.debug(f"Resolved model alias '{model_path}' -> '{resolved_path}'")

    # Resolve container alias
    container = model.get('container', '')

    if 'containers' in cluster_config and container in cluster_config['containers']:
        resolved_container = cluster_config['containers'][container]
        model['container'] = resolved_container
        logging.debug(f"Resolved container alias '{container}' -> '{resolved_container}'")
    elif 'container' not in model and 'default_container' in cluster_config:
        model['container'] = cluster_config['default_container']
        logging.debug(f"Applied default container: {model['container']}")

    return config


def get_srtslurm_setting(key: str, default=None):
    """
    Get a setting from srtslurm.yaml cluster config.

    Args:
        key: Setting key (e.g., 'gpus_per_node', 'network_interface')
        default: Default value if not found

    Returns:
        Setting value or default if not found
    """
    cluster_config = load_cluster_config()
    if cluster_config and key in cluster_config:
        return cluster_config[key]
    return default


def load_config(path: Path) -> dict:
    """
    Load and validate YAML config, applying cluster defaults.

    Returns fully resolved config ready for submission.
    """
    from .validation import validate_config

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Load user config
    with open(path) as f:
        user_config = yaml.safe_load(f)

    # Load cluster defaults (optional)
    cluster_config = load_cluster_config()

    # Resolve with defaults
    config = resolve_config_with_defaults(user_config, cluster_config)

    # Validate
    errors = validate_config(config)
    if errors:
        raise ValueError(
            f"Invalid config in {path}:\n  " + "\n  ".join(errors)
        )

    logging.info(f"Loaded config: {config['name']}")
    return config
