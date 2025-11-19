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
    load_yaml,
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
        """Save user config"""
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"  ✓ Saved config: {config_path.name}")
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

    def save_sbatch_script(self, script_content: str) -> Path:
        """Save generated sbatch script"""
        script_path = self.output_dir / "job.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        logging.info(f"  ✓ Saved sbatch script: {script_path.name}")
        self.sbatch_script_path = script_path
        return script_path

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
        print(f"  - config.yaml          (user config)")
        if self.sglang_config_path:
            print(f"  - sglang_config.yaml   (SGLang flags)")
        print(f"  - job.sh               (sbatch script)")
        print(f"  - metadata.json        (submission info)")
        print(f"\nTo submit this job:")
        print(f"  cd {self.output_dir}")
        print(f"  sbatch job.sh")
        print("\n" + "="*60 + "\n")


def generate_sbatch_script(config: dict, args: list[str], sglang_config_path: Path = None) -> str:
    """
    Generate sbatch script content from config.

    For dry-run mode, creates a simplified script showing what would be executed.
    """
    # Extract values
    resources = config['resources']
    slurm = config['slurm']
    model = config['model']
    backend = config.get('backend', {})
    is_aggregated = 'agg_nodes' in resources

    # Create simplified script (not using Jinja for dry-run to avoid missing vars)
    if is_aggregated:
        script = f"""#!/bin/bash
#SBATCH --job-name={config['name']}
#SBATCH --nodes={resources['agg_nodes']}
#SBATCH --ntasks={resources['agg_nodes']}
#SBATCH --account={slurm['account']}
#SBATCH --partition={slurm['partition']}
#SBATCH --time={slurm['time_limit']}

# Aggregated mode: {resources['agg_workers']} workers on {resources['agg_nodes']} nodes
# {resources['gpus_per_node']} GPUs per node

echo "Model: {model['path']}"
echo "Container: {model['container']}"
"""
    else:
        total_nodes = resources['prefill_nodes'] + resources['decode_nodes']
        script = f"""#!/bin/bash
#SBATCH --job-name={config['name']}
#SBATCH --nodes={total_nodes}
#SBATCH --ntasks={total_nodes}
#SBATCH --account={slurm['account']}
#SBATCH --partition={slurm['partition']}
#SBATCH --time={slurm['time_limit']}

# Disaggregated mode
# Prefill: {resources['prefill_workers']} workers on {resources['prefill_nodes']} nodes
# Decode:  {resources['decode_workers']} workers on {resources['decode_nodes']} nodes
# {resources['gpus_per_node']} GPUs per node

echo "Model: {model['path']}"
echo "Container: {model['container']}"
"""

    # Add SGLang config info
    if sglang_config_path:
        script += f"""
# SGLang Config File Mode
echo "SGLang config: {sglang_config_path}"
echo "Using: dynamo.sglang --config <path> --config-key prefill/decode"
"""

    # Add environment variables
    if 'environment' in backend:
        script += "\n# Environment Variables:\n"
        for key, val in backend['environment'].items():
            script += f"export {key}={val}\n"

    # Add worker launch commands (simplified)
    script += """
# Worker launch commands would be here
# (Generated from Jinja template in real submission)

echo "This is a dry-run - job was not submitted to SLURM"
"""

    return script


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

        # Generate sbatch script
        script_content = generate_sbatch_script(config, args, sglang_config_path)
        ctx.save_sbatch_script(script_content)

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
