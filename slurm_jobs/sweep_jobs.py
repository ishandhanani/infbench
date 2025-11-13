#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parameter sweep orchestrator for SGLang benchmarking.

This script reads a YAML configuration with arrays of parameter values,
generates a cartesian product of all combinations, and submits multiple
SLURM jobs via submit_job_script.py for performance comparison.
"""

import argparse
import itertools
import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import Any

import yaml


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s| %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_sweep_config(config_path: str) -> dict[str, dict[str, list]]:
    """Parse the YAML sweep configuration file.
    
    Returns:
        Dictionary with keys 'slurm' and 'sglang',
        each containing parameter names mapped to lists of values.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ValueError(f"Empty or invalid YAML config: {config_path}")
    
    # Validate structure
    valid_sections = {"slurm", "sglang"}
    for section in config:
        if section not in valid_sections:
            raise ValueError(f"Invalid section '{section}' in config. Must be one of: {valid_sections}")
    
    # Ensure all values are lists
    for section in config:
        for param, values in config[section].items():
            if not isinstance(values, list):
                config[section][param] = [values]
    
    return config


def generate_combinations(config: dict[str, dict[str, list]]) -> list[dict[str, Any]]:
    """Generate cartesian product of all parameter combinations.
    
    Returns:
        List of dictionaries, each representing one complete parameter combination.
    """
    combinations = []
    
    # Separate sections
    slurm_params = config.get("slurm", {})
    sglang_params = config.get("sglang", {})
    
    # Generate cartesian products for each section
    slurm_combos = [{}]
    if slurm_params:
        keys = list(slurm_params.keys())
        values = [slurm_params[k] for k in keys]
        slurm_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    sglang_combos = [{}]
    if sglang_params:
        keys = list(sglang_params.keys())
        values = [sglang_params[k] for k in keys]
        sglang_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    # Combine all sections
    for slurm, sglang in itertools.product(slurm_combos, sglang_combos):
        combinations.append({
            "slurm": slurm,
            "sglang": sglang,
        })
    
    return combinations


def create_sglang_config(
    sglang_params: dict[str, Any], 
    output_path: str,
    is_disaggregated: bool = False
) -> None:
    """Create a YAML config file for sglang with the given parameters.
    
    For disaggregated mode, creates nested YAML with 'prefill' and 'decode' keys.
    For aggregated mode, creates flat YAML.
    """
    if is_disaggregated:
        # Create nested structure for disagg mode
        config = {
            "prefill": sglang_params.copy(),
            "decode": sglang_params.copy(),
        }
    else:
        # Flat structure for aggregated mode
        config = sglang_params
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def submit_job(
    combo: dict[str, Any],
    combo_idx: int,
    args: argparse.Namespace,
    sglang_config_path: str | None,
    output_dir: pathlib.Path,
    is_disaggregated: bool,
) -> str:
    """Submit a single job for the given parameter combination.
    
    Returns:
        The job ID of the submitted job.
    """
    slurm_params = combo["slurm"]
    
    # Build command for submit_job_script.py
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).parent / "submit_job_script.py"),
        "--account", args.account,
        "--partition", args.partition,
        "--model-dir", args.model_dir,
        "--config-dir", args.config_dir,
        "--container-image", args.container_image,
        "--gpu-type", args.gpu_type,
        "--script-variant", args.script_variant,
    ]
    
    # Add optional global arguments
    if args.time_limit:
        cmd.extend(["--time-limit", args.time_limit])
    if args.gpus_per_node:
        cmd.extend(["--gpus-per-node", str(args.gpus_per_node)])
    if args.network_interface:
        cmd.extend(["--network-interface", args.network_interface])
    if args.job_name:
        cmd.extend(["--job-name", f"{args.job_name}_sweep{combo_idx}"])
    if args.log_dir:
        cmd.extend(["--log-dir", args.log_dir])
    if args.use_init_location:
        cmd.append("--use-init-location")
    if args.disable_config_dump:
        cmd.append("--disable-config-dump")
    if args.use_dynamo_whls:
        cmd.append("--use-dynamo-whls")
    if args.enable_multiple_frontends:
        cmd.append("--enable-multiple-frontends")
        if args.num_additional_frontends:
            cmd.extend(["--num-additional-frontends", str(args.num_additional_frontends)])
    if args.profiler:
        cmd.extend(["--profiler", args.profiler])
    if args.retries:
        cmd.extend(["--retries", str(args.retries)])
    
    # Add SLURM parameters from sweep config
    if "prefill_nodes" in slurm_params:
        cmd.extend(["--prefill-nodes", str(slurm_params["prefill_nodes"])])
    if "decode_nodes" in slurm_params:
        cmd.extend(["--decode-nodes", str(slurm_params["decode_nodes"])])
    if "prefill_workers" in slurm_params:
        cmd.extend(["--prefill-workers", str(slurm_params["prefill_workers"])])
    if "decode_workers" in slurm_params:
        cmd.extend(["--decode-workers", str(slurm_params["decode_workers"])])
    if "agg_nodes" in slurm_params:
        cmd.extend(["--agg-nodes", str(slurm_params["agg_nodes"])])
    if "agg_workers" in slurm_params:
        cmd.extend(["--agg-workers", str(slurm_params["agg_workers"])])
    
    # Add sglang config if present
    if sglang_config_path:
        cmd.extend(["--sglang-config", sglang_config_path])
    
    # Log the command
    logging.info(f"Submitting job {combo_idx + 1} with parameters:")
    logging.info(f"  SLURM: {slurm_params}")
    logging.info(f"  SGLang config: {sglang_config_path}")
    
    # Execute submission
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Parse job ID from output (submit_job_script.py logs it)
        # Look for pattern like "Job submitted successfully with ID: 12345"
        for line in result.stderr.split("\n"):
            if "Job submitted successfully with ID:" in line:
                job_id = line.split(":")[-1].strip()
                logging.info(f"Job {combo_idx + 1} submitted with ID: {job_id}")
                return job_id
        
        # Fallback: just return a placeholder
        logging.warning(f"Could not parse job ID from output for combo {combo_idx + 1}")
        return f"unknown_{combo_idx}"
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to submit job {combo_idx + 1}: {e}")
        logging.error(f"stderr: {e.stderr}")
        raise


def save_sweep_manifest(
    combinations: list[dict[str, Any]],
    job_ids: list[str],
    output_path: pathlib.Path,
) -> None:
    """Save a manifest of all submitted jobs and their parameters."""
    manifest = {
        "sweep_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_jobs": len(combinations),
        "jobs": [
            {
                "job_id": job_id,
                "parameters": combo,
            }
            for job_id, combo in zip(job_ids, combinations)
        ],
    }
    
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logging.info(f"Saved sweep manifest to {output_path}")


def _parse_command_line_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep parameter configurations and submit multiple SLURM jobs"
    )
    
    # Required arguments
    parser.add_argument(
        "--sweep-config",
        required=True,
        help="Path to YAML sweep configuration file",
    )
    parser.add_argument("--account", required=True, help="SLURM account")
    parser.add_argument("--partition", required=True, help="SLURM partition")
    parser.add_argument("--model-dir", required=True, help="Model directory path")
    parser.add_argument("--config-dir", required=True, help="Config directory path")
    parser.add_argument("--container-image", required=True, help="Container image")
    parser.add_argument("--gpu-type", required=True, help="GPU type to use")
    parser.add_argument(
        "--script-variant",
        required=True,
        help="Script variant to use (e.g., 'max-tpt', '1p_4d')",
    )
    
    # Optional arguments that pass through to submit_job_script.py
    parser.add_argument("--job-name", default="sweep_job", help="Base job name")
    parser.add_argument("--time-limit", default="04:00:00", help="Time limit (HH:MM:SS)")
    parser.add_argument("--gpus-per-node", type=int, default=8, help="GPUs per node")
    parser.add_argument("--network-interface", default="eth3", help="Network interface")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory")
    parser.add_argument("--use-init-location", action="store_true", help="Use init locations")
    parser.add_argument("--disable-config-dump", action="store_true", help="Disable config dump")
    parser.add_argument("--use-dynamo-whls", action="store_true", help="Use dynamo wheels")
    parser.add_argument("--enable-multiple-frontends", action="store_true", help="Enable multiple frontends")
    parser.add_argument("--num-additional-frontends", type=int, default=0, help="Number of additional frontends")
    parser.add_argument("--profiler", type=str, help="Profiler configurations")
    parser.add_argument("--retries", type=int, default=0, help="Number of retries")
    
    # Sweep-specific arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save sweep configs and manifest (default: slurm_jobs/sweep_<timestamp>)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print combinations without submitting jobs",
    )
    
    return parser.parse_args(args)


def main(input_args: list[str] | None = None):
    setup_logging()
    args = _parse_command_line_args(input_args)
    
    # Parse sweep configuration
    logging.info(f"Parsing sweep config: {args.sweep_config}")
    config = parse_sweep_config(args.sweep_config)
    
    # Generate combinations
    combinations = generate_combinations(config)
    logging.info(f"Generated {len(combinations)} parameter combinations")
    
    if args.dry_run:
        logging.info("DRY RUN - printing combinations:")
        for i, combo in enumerate(combinations):
            print(f"\nCombination {i + 1}:")
            print(f"  SLURM: {combo['slurm']}")
            print(f"  SGLang: {combo['sglang']}")
        return
    
    # Create output directory for sweep artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    else:
        output_dir = pathlib.Path(__file__).parent / f"sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Sweep output directory: {output_dir}")
    
    # Determine if disaggregated mode based on first combination
    is_disaggregated = False
    if combinations:
        first_slurm = combinations[0].get("slurm", {})
        is_disaggregated = "prefill_nodes" in first_slurm or "decode_nodes" in first_slurm
    
    # Submit jobs
    job_ids = []
    for i, combo in enumerate(combinations):
        # Create sglang config file if sglang params exist
        sglang_config_path = None
        if combo["sglang"]:
            sglang_config_path = str(output_dir / f"sglang_config_{i}.yaml")
            create_sglang_config(combo["sglang"], sglang_config_path, is_disaggregated)
            logging.info(f"Created sglang config: {sglang_config_path}")
        
        # Submit job
        try:
            job_id = submit_job(combo, i, args, sglang_config_path, output_dir, is_disaggregated)
            job_ids.append(job_id)
        except Exception as e:
            logging.error(f"Failed to submit job {i + 1}: {e}")
            job_ids.append(f"failed_{i}")
    
    # Save manifest
    manifest_path = output_dir / "sweep_manifest.json"
    save_sweep_manifest(combinations, job_ids, manifest_path)
    
    # Print summary
    print("\n" + "="*80)
    print(f"🚀 Parameter Sweep Complete!")
    print("="*80)
    print(f"Total combinations: {len(combinations)}")
    print(f"Jobs submitted: {len([j for j in job_ids if not j.startswith('failed')])}")
    print(f"Failed submissions: {len([j for j in job_ids if j.startswith('failed')])}")
    print(f"\nSweep artifacts saved to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print("\nSubmitted job IDs:")
    for i, job_id in enumerate(job_ids):
        status = "✓" if not job_id.startswith("failed") else "✗"
        print(f"  {status} Combination {i + 1}: {job_id}")
    print("="*80)


if __name__ == "__main__":
    main()

