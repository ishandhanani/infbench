#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified job submission interface for InfBench.

This is the single entrypoint for submitting benchmarks, replacing both
submit_job_script.py and submit_yaml.py with a cleaner, unified interface.

Inspired by ignition's clean CLI design.

Usage:
    # Submit from YAML config
    python submit.py config.yaml

    # Submit from YAML with sweep
    python submit.py sweep.yaml --sweep

    # Dry-run mode (validate without submitting)
    python submit.py config.yaml --dry-run

    # Legacy command-line mode (backward compatible)
    python submit.py --prefill-nodes 1 --decode-nodes 12 ...

Features:
    - Single entrypoint for all submission modes
    - YAML config support with SGLang config generation
    - Parameter sweeping (grid and list)
    - Dry-run mode for validation and testing
    - Backward compatible with legacy CLI args
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

# Import config generation logic
from submit_yaml import (
    generate_sglang_config_file,
    yaml_to_args,
    generate_sweep_configs,
    expand_template,
    format_benchmark,
)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
                # Aggregated needs 'shared' or 'aggregated' section
                if 'shared' not in sglang_cfg and 'aggregated' not in sglang_cfg:
                    errors.append("Aggregated mode requires backend.sglang_config.shared or backend.sglang_config.aggregated")
            else:
                # Disaggregated needs prefill and decode sections
                for mode in ['prefill', 'decode']:
                    if mode not in sglang_cfg:
                        errors.append(f"Disaggregated mode requires backend.sglang_config.{mode}")

    return errors


def load_cluster_config() -> dict | None:
    """
    Load cluster configuration from srtslurm.yaml if it exists.

    Returns None if file doesn't exist (graceful degradation).
    """
    cluster_config_path = Path(__file__).parent.parent / "srtslurm.yaml"

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
    import copy
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


def load_yaml(path: Path) -> dict:
    """
    Load and validate YAML config, applying cluster defaults.

    Returns fully resolved config ready for submission.
    """
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


class DryRunContext:
    """Context for dry-run mode - creates output directory and saves artifacts"""

    def __init__(self, config: dict, job_name: str = None):
        self.config = config
        self.job_name = job_name or config.get('name', 'dry-run')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = None
        self.sglang_config_path = None
        self.sbatch_script_path = None

    def setup(self) -> Path:
        """Create dry-run output directory"""
        # Create in ../dry-runs/
        base_dir = Path(__file__).parent.parent / "dry-runs"
        self.output_dir = base_dir / f"{self.job_name}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"📁 Dry-run output directory: {self.output_dir}")
        return self.output_dir

    def save_config(self, config: dict) -> Path:
        """Save resolved config (with all defaults applied)"""
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logging.info(f"  ✓ Saved resolved config: {config_path.name}")
        return config_path

    def save_sglang_config(self, sglang_config_path: Path) -> Path:
        """Copy SGLang config to dry-run dir"""
        if sglang_config_path and sglang_config_path.exists():
            dest = self.output_dir / "sglang_config.yaml"
            shutil.copy(sglang_config_path, dest)
            logging.info(f"  ✓ Saved SGLang config: {dest.name}")
            self.sglang_config_path = dest
            return dest
        return None

    def save_rendered_commands(self, config: dict, sglang_config_path: Path) -> Path:
        """Save just the rendered commands (no sbatch headers)"""
        commands_path = self.output_dir / "commands.sh"

        content = "#!/bin/bash\n"
        content += "# Generated SGLang commands\n"
        content += f"# Config: {sglang_config_path}\n\n"
        content += "# ============================================================\n"
        content += "# PREFILL WORKER COMMAND\n"
        content += "# ============================================================\n\n"
        content += render_sglang_command(config, sglang_config_path, mode="prefill")
        content += "\n\n"
        content += "# ============================================================\n"
        content += "# DECODE WORKER COMMAND\n"
        content += "# ============================================================\n\n"
        content += render_sglang_command(config, sglang_config_path, mode="decode")
        content += "\n"

        with open(commands_path, 'w') as f:
            f.write(content)
        commands_path.chmod(0o755)
        logging.info(f"  ✓ Saved rendered commands: {commands_path.name}")
        return commands_path

    def save_metadata(self, config: dict, args: list[str]) -> Path:
        """Save submission metadata"""
        metadata = {
            "job_name": self.job_name,
            "timestamp": self.timestamp,
            "config": config,
            "submit_args": args,
            "mode": "dry-run",
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"  ✓ Saved metadata: {metadata_path.name}")
        return metadata_path

    def print_summary(self):
        """Print summary of what would be submitted"""
        print("\n" + "="*60)
        print("🔍 DRY-RUN SUMMARY")
        print("="*60)
        print(f"\nJob Name: {self.job_name}")
        print(f"Output Directory: {self.output_dir}")
        print(f"\nGenerated Files:")
        print(f"  - config.yaml          (resolved config with defaults)")
        if self.sglang_config_path:
            print(f"  - sglang_config.yaml   (SGLang flags)")
        print(f"  - commands.sh          (full bash commands)")
        print(f"  - metadata.json        (submission info)")
        print(f"\nTo see what commands would run:")
        print(f"  cat {self.output_dir}/commands.sh")
        print("\n" + "="*60 + "\n")


def render_sglang_command(config: dict, sglang_config_path: Path, mode: str = "prefill") -> str:
    """
    Render the full SGLang command that would be executed with all flags inlined.

    Args:
        config: User config dict
        sglang_config_path: Path to generated SGLang config
        mode: "prefill" or "decode"

    Returns:
        Multi-line string showing the full command with environment variables and all flags
    """
    backend = config.get('backend', {})
    resources = config.get('resources', {})

    # Environment variables
    env_vars = []
    if 'environment' in backend:
        for key, val in backend['environment'].items():
            env_vars.append(f"{key}={val}")

    # Add decode-specific env vars if in decode mode
    if mode == "decode" and 'decode_environment' in backend:
        for key, val in backend['decode_environment'].items():
            env_vars.append(f"{key}={val}")

    # Build command
    lines = []

    # Environment variables (one per line with backslash continuation)
    if env_vars:
        for env_var in env_vars:
            lines.append(f"{env_var} \\")

    # Python command
    lines.append("python3 -m dynamo.sglang \\")

    # Load the generated SGLang config and inline all flags
    with open(sglang_config_path) as f:
        sglang_config = yaml.load(f, Loader=yaml.FullLoader)

    # Get flags for this mode
    mode_config = sglang_config.get(mode, {})

    # Convert config dict to command-line flags
    for key, value in sorted(mode_config.items()):
        # Convert underscores to hyphens for CLI flags
        flag_name = key.replace('_', '-')

        # Handle different value types
        if isinstance(value, bool):
            if value:
                lines.append(f"    --{flag_name} \\")
        elif isinstance(value, list):
            # For lists, pass each value as separate argument
            values_str = " ".join(str(v) for v in value)
            lines.append(f"    --{flag_name} {values_str} \\")
        else:
            lines.append(f"    --{flag_name} {value} \\")

    # Determine nnodes based on mode and whether aggregated or disaggregated
    is_aggregated = 'agg_nodes' in resources
    if is_aggregated:
        nnodes = resources['agg_nodes']
    else:
        # Disaggregated: prefill and decode have different node counts
        if mode == "prefill":
            nnodes = resources['prefill_nodes']
        else:  # decode
            nnodes = resources['decode_nodes']

    # Coordination flags
    lines.append("    --dist-init-addr $HOST_IP_MACHINE:$PORT \\")
    lines.append(f"    --nnodes {nnodes} \\")
    lines.append("    --node-rank $RANK \\")

    # Parallelism flags (computed from resources)
    gpus_per_node = resources.get('gpus_per_node', 4)
    lines.append(f"    --ep-size {gpus_per_node} \\")
    lines.append(f"    --tp-size {gpus_per_node} \\")
    lines.append(f"    --dp-size {gpus_per_node}")

    return "\n".join(lines)


def submit_single(config_path: Path = None, config: dict = None, dry_run: bool = False):
    """
    Submit a single job from YAML config.

    Args:
        config_path: Path to YAML config file (or None if config provided)
        config: Pre-loaded config dict (or None if loading from path)
        dry_run: If True, don't submit to SLURM, just validate and save artifacts
    """
    # Load config if needed
    if config is None:
        config = load_yaml(config_path)

    # Dry-run mode
    if dry_run:
        logging.info(f"🔍 DRY-RUN MODE: {config['name']}")
        ctx = DryRunContext(config)
        ctx.setup()

        # Save user config
        ctx.save_config(config)

        # Generate SGLang config if needed
        sglang_config_path = None
        if config.get('backend', {}).get('type') == 'sglang':
            sglang_config_path = generate_sglang_config_file(config)
            ctx.save_sglang_config(sglang_config_path)

        # Convert to args (for metadata)
        args = yaml_to_args(config, sglang_config_path)

        # Save rendered commands (full bash commands with all flags inlined)
        if sglang_config_path:
            ctx.save_rendered_commands(config, sglang_config_path)

        # Save metadata
        ctx.save_metadata(config, args)

        # Print summary
        ctx.print_summary()

        return

    # Real submission mode
    logging.info(f"🚀 Submitting job: {config['name']}")

    # Generate SGLang config if needed
    sglang_config_path = None
    if config.get('backend', {}).get('type') == 'sglang':
        sglang_config_path = generate_sglang_config_file(config)

    # Convert to args
    args = yaml_to_args(config, sglang_config_path)

    # Call existing submit_job_script
    from submit_job_script import main as submit_job
    submit_job(args)


def submit_sweep(config_path: Path, dry_run: bool = False):
    """
    Submit parameter sweep.

    Args:
        config_path: Path to sweep YAML config
        dry_run: If True, don't submit to SLURM, just validate and save artifacts
    """
    sweep_config = load_yaml(config_path)

    # Generate all configs
    configs = generate_sweep_configs(sweep_config)
    logging.info(f"Generated {len(configs)} configurations for sweep")

    if dry_run:
        logging.info(f"🔍 DRY-RUN MODE: Sweep with {len(configs)} jobs")

        # Create sweep output directory
        sweep_dir = Path(__file__).parent.parent / "dry-runs" / f"{sweep_config['name']}_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"📁 Sweep directory: {sweep_dir}")

        # Save sweep config
        with open(sweep_dir / "sweep_config.yaml", 'w') as f:
            yaml.dump(sweep_config, f, default_flow_style=False)

        # Generate each job
        for i, (config, params) in enumerate(configs, 1):
            logging.info(f"\n[{i}/{len(configs)}] {config['name']}")
            logging.info(f"  Parameters: {params}")

            # Create job directory
            job_dir = sweep_dir / f"job_{i:03d}_{config['name']}"
            job_dir.mkdir(exist_ok=True)

            # Save config
            with open(job_dir / "config.yaml", 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            # Generate SGLang config
            if config.get('backend', {}).get('type') == 'sglang':
                sglang_config_path = generate_sglang_config_file(config, params)
                if sglang_config_path:
                    shutil.copy(sglang_config_path, job_dir / "sglang_config.yaml")

            logging.info(f"  ✓ Saved to: {job_dir.name}")

        print("\n" + "="*60)
        print(f"🔍 SWEEP DRY-RUN SUMMARY")
        print("="*60)
        print(f"\nSweep: {sweep_config['name']}")
        print(f"Jobs: {len(configs)}")
        print(f"Output: {sweep_dir}")
        print(f"\nEach job directory contains:")
        print(f"  - config.yaml (expanded config)")
        print(f"  - sglang_config.yaml (if applicable)")
        print("\n" + "="*60 + "\n")

        return

    # Real submission
    for i, (config, params) in enumerate(configs, 1):
        logging.info(f"\n[{i}/{len(configs)}] Submitting: {config['name']}")
        logging.info(f"  Parameters: {params}")
        submit_single(config=config, dry_run=False)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Unified job submission for InfBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit from YAML config
  python submit.py config.yaml

  # Submit sweep
  python submit.py sweep.yaml --sweep

  # Dry-run (validate without submitting)
  python submit.py config.yaml --dry-run

  # Dry-run sweep (generate all configs without submitting)
  python submit.py sweep.yaml --sweep --dry-run

  # Legacy CLI mode (backward compatible)
  python submit.py --prefill-nodes 1 --decode-nodes 12 ...
        """
    )

    # Primary argument: config file or legacy mode
    parser.add_argument(
        "config",
        type=Path,
        nargs='?',
        help="YAML config file (omit for legacy CLI mode)"
    )

    # Mode flags
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Treat as sweep config (multiple jobs)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and generate artifacts without submitting to SLURM"
    )

    # Legacy mode args (for backward compatibility)
    # ... (would add all the existing submit_job_script.py args here)

    args = parser.parse_args()

    # Determine mode
    if args.config:
        # YAML mode
        if not args.config.exists():
            logging.error(f"Config file not found: {args.config}")
            sys.exit(1)

        try:
            if args.sweep:
                submit_sweep(args.config, dry_run=args.dry_run)
            else:
                submit_single(args.config, dry_run=args.dry_run)
        except Exception as e:
            logging.exception(f"Error: {e}")
            sys.exit(1)
    else:
        # Legacy CLI mode
        logging.error("Legacy CLI mode not yet implemented in unified interface")
        logging.error("Please use YAML config or use submit_job_script.py directly")
        sys.exit(1)


if __name__ == "__main__":
    main()
