#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified job submission interface for srtctl.

This is the main entrypoint for submitting benchmarks via YAML configs.

Usage:
    srtctl config.yaml
    srtctl config.yaml --dry-run
    srtctl sweep.yaml --sweep
"""

import argparse
import json
import logging
import os
import shutil
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Import from srtctl modules
from srtctl.core.config import load_config
from srtctl.backends.sglang import SGLangBackend


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

    def setup(self) -> Path:
        """Create dry-run output directory"""
        # Create in dry-runs/
        base_dir = Path.cwd() / "dry-runs"
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

    def save_rendered_commands(self, backend, sglang_config_path: Path) -> Path:
        """Save just the rendered commands (no sbatch headers)"""
        commands_path = self.output_dir / "commands.sh"

        content = "#!/bin/bash\n"
        content += "# Generated SGLang commands\n"
        content += f"# Config: {sglang_config_path}\n\n"
        content += "# ============================================================\n"
        content += "# PREFILL WORKER COMMAND\n"
        content += "# ============================================================\n\n"
        content += backend.render_command(mode="prefill", config_path=sglang_config_path)
        content += "\n\n"
        content += "# ============================================================\n"
        content += "# DECODE WORKER COMMAND\n"
        content += "# ============================================================\n\n"
        content += backend.render_command(mode="decode", config_path=sglang_config_path)
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
        config = load_config(config_path)

    # Dry-run mode
    if dry_run:
        logging.info(f"🔍 DRY-RUN MODE: {config['name']}")
        ctx = DryRunContext(config)
        ctx.setup()

        # Save user config
        ctx.save_config(config)

        # Create backend instance
        backend_type = config.get('backend', {}).get('type')
        if backend_type == 'sglang':
            backend = SGLangBackend(config)
            sglang_config_path = backend.generate_config_file()
            ctx.save_sglang_config(sglang_config_path)

            # Save rendered commands
            if sglang_config_path:
                ctx.save_rendered_commands(backend, sglang_config_path)
        else:
            sglang_config_path = None

        # Save metadata (no more args conversion needed)
        ctx.save_metadata(config, [])

        # Print summary
        ctx.print_summary()

        return

    # Real submission mode
    logging.info(f"🚀 Submitting job: {config['name']}")

    # Create backend and generate config
    backend_type = config.get('backend', {}).get('type')
    if backend_type == 'sglang':
        backend = SGLangBackend(config)
        sglang_config_path = backend.generate_config_file()

        # Generate SLURM job script using backend
        import subprocess
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path, rendered_script = backend.generate_slurm_script(
            config_path=sglang_config_path,
            timestamp=timestamp
        )

        # Submit to SLURM
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse job ID from sbatch output
            job_id = result.stdout.strip().split()[-1]
            logging.info(f"✅ Job submitted successfully with ID: {job_id}")

            # Create log directory
            is_aggregated = 'agg_nodes' in config.get('resources', {})
            if is_aggregated:
                agg_workers = config['resources']['agg_workers']
                log_dir_name = f"{job_id}_{agg_workers}A_{timestamp}"
            else:
                prefill_workers = config['resources']['prefill_workers']
                decode_workers = config['resources']['decode_workers']
                log_dir_name = f"{job_id}_{prefill_workers}P_{decode_workers}D_{timestamp}"

            # Create log directory in infbench repo (parent of infbench-yaml-config)
            yaml_config_root = Path(__file__).parent.parent.parent.parent
            infbench_root = yaml_config_root.parent / "infbench"
            log_dir = infbench_root / "logs" / log_dir_name
            log_dir.mkdir(parents=True, exist_ok=True)

            # Save rendered script
            with open(log_dir / "sbatch_script.sh", 'w') as f:
                f.write(rendered_script)

            # Save config
            import yaml
            with open(log_dir / "config.yaml", 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            # Save SGLang config if present
            if sglang_config_path:
                import shutil
                shutil.copy(sglang_config_path, log_dir / "sglang_config.yaml")

            logging.info(f"📁 Logs directory: {log_dir}")
            print(f"\n✅ Job {job_id} submitted!")
            print(f"📁 Logs: {log_dir}\n")

        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Error submitting job: {e}")
            logging.error(f"stderr: {e.stderr}")
            raise
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def submit_sweep(config_path: Path, dry_run: bool = False):
    """
    Submit parameter sweep.

    Args:
        config_path: Path to sweep YAML config
        dry_run: If True, don't submit to SLURM, just validate and save artifacts
    """
    # Import sweep logic from submit_yaml
    import sys
    scripts_path = Path(__file__).parent.parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_path))

    from submit_yaml import generate_sweep_configs

    sweep_config = load_config(config_path)

    # Generate all configs
    configs = generate_sweep_configs(sweep_config)
    logging.info(f"Generated {len(configs)} configurations for sweep")

    if dry_run:
        logging.info(f"🔍 DRY-RUN MODE: Sweep with {len(configs)} jobs")

        # Create sweep output directory
        sweep_dir = Path.cwd() / "dry-runs" / f"{sweep_config['name']}_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        description="Unified job submission for srtctl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit from YAML config
  srtctl config.yaml

  # Submit sweep
  srtctl sweep.yaml --sweep

  # Dry-run (validate without submitting)
  srtctl config.yaml --dry-run

  # Dry-run sweep (generate all configs without submitting)
  srtctl sweep.yaml --sweep --dry-run
        """
    )

    # Primary argument: config file
    parser.add_argument(
        "config",
        type=Path,
        help="YAML config file"
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

    args = parser.parse_args()

    # Check config exists
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


if __name__ == "__main__":
    main()
