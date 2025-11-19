#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Config validation logic.
"""


def validate_config(config: dict) -> list[str]:
    """
    Validate config structure and return list of errors.

    Returns empty list if valid, otherwise list of error messages.
    """
    errors = []

    # Required top-level keys
    required_keys = ['name', 'slurm', 'resources', 'model', 'backend']
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required top-level key: '{key}'")

    # Validate slurm config
    if 'slurm' in config:
        slurm = config['slurm']
        for key in ['account', 'partition']:
            if key not in slurm:
                errors.append(f"Missing required slurm.{key}")

    # Validate resources (either aggregated or disaggregated)
    if 'resources' in config:
        resources = config['resources']

        # Check if aggregated or disaggregated
        is_aggregated = 'agg_nodes' in resources

        if is_aggregated:
            # Aggregated mode validation
            for key in ['agg_nodes', 'agg_workers', 'gpus_per_node']:
                if key not in resources:
                    errors.append(f"Aggregated mode requires resources.{key}")
        else:
            # Disaggregated mode validation
            for key in ['prefill_nodes', 'decode_nodes', 'prefill_workers', 'decode_workers', 'gpus_per_node']:
                if key not in resources:
                    errors.append(f"Disaggregated mode requires resources.{key}")

    # Validate model config
    if 'model' in config:
        model = config['model']
        for key in ['path', 'container']:
            if key not in model:
                errors.append(f"Missing required model.{key}")

    # Validate backend config
    if 'backend' in config:
        backend = config['backend']
        if 'type' not in backend:
            errors.append("Missing required backend.type")

        # If sglang backend, validate sglang_config
        if backend.get('type') == 'sglang' and 'sglang_config' in backend:
            sglang_cfg = backend['sglang_config']

            # Check for either aggregated or disaggregated config
            is_aggregated = 'agg_nodes' in config.get('resources', {})

            if is_aggregated:
                # Aggregated needs 'aggregated' section
                if 'aggregated' not in sglang_cfg:
                    errors.append("Aggregated mode requires backend.sglang_config.aggregated")
            else:
                # Disaggregated needs prefill and decode sections
                for mode in ['prefill', 'decode']:
                    if mode not in sglang_cfg:
                        errors.append(f"Disaggregated mode requires backend.sglang_config.{mode}")

    return errors
