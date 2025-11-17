# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to generate SLURM job scripts from Jinja2 templates.
"""

import argparse
import json
import logging
import os
import pathlib
import subprocess
import tempfile
from datetime import datetime

from jinja2 import Template

from cluster_config import validate_cluster_settings, get_cluster_setting


def print_welcome_message(job_ids: list[str], log_dir_name: str):
    """Print a concise welcome message with log directory info."""
    print(
        f"\nYour logs will be in ../logs/{log_dir_name}. To access them, run:\n\n    cd ../logs/{log_dir_name}\n"
    )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s| %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_job_script(template_path, output_path, **kwargs):
    """Generate a job script from template with given parameters."""
    with open(template_path, "r") as f:
        template = Template(f.read())

    rendered_script = template.render(**kwargs)
    with open(output_path, "w") as f:
        f.write(rendered_script)

    return output_path, rendered_script


def submit_job(job_script_path, extra_slurm_args=[]):
    """
    Submit the job script to SLURM and extract the job ID from the output.

    Returns:
        The job ID of the submitted job.
    """
    try:
        command = (
            ["sbatch"]
            + ["--" + x for x in extra_slurm_args]
            + [
                job_script_path,
            ]
        )
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_lines = result.stdout.strip().split("\n")

        # sbatch typically outputs: "Submitted batch job JOBID"
        job_id = output_lines[-1].split()[-1]
        logging.info(f"Job submitted successfully with ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logging.error(f"Error submitting job: {e}")
        logging.error(f"stderr: {e.stderr}")
        raise
    except (IndexError, ValueError):
        logging.error(f"Error parsing job ID from sbatch output: {result.stdout}")
        raise


def create_job_metadata(
    job_id: str,
    timestamp: str,
    args: argparse.Namespace,
    benchmark_config: dict,
):
    """
    Create job metadata dictionary from job submission args.

    Returns:
        Dictionary with job metadata.
    """
    metadata = {
        "version": "1.0",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_metadata": {
            "slurm_job_id": job_id,
            "run_date": timestamp,
            "job_name": args.job_name,
            "account": args.account,
            "partition": args.partition,
            "time_limit": args.time_limit,
            "container": args.container_image,
            "model_dir": args.model_dir,
            "config_dir": args.config_dir,
            "gpus_per_node": args.gpus_per_node,
            "network_interface": args.network_interface,
            "gpu_type": args.gpu_type,
            "script_variant": args.script_variant,
            "use_init_location": args.use_init_location,
            "enable_config_dump": args.enable_config_dump,
            "use_dynamo_whls": True,  # Always true when config-dir is set
            "log_dir": args.log_dir if args.log_dir else "logs",
        },
        "profiler_metadata": benchmark_config,
    }

    # Add mode-specific metadata
    if args.agg_nodes is not None:
        metadata["run_metadata"]["mode"] = "aggregated"
        metadata["run_metadata"]["agg_nodes"] = args.agg_nodes
        metadata["run_metadata"]["agg_workers"] = args.agg_workers
    else:
        metadata["run_metadata"]["mode"] = "disaggregated"
        metadata["run_metadata"]["prefill_nodes"] = args.prefill_nodes
        metadata["run_metadata"]["decode_nodes"] = args.decode_nodes
        metadata["run_metadata"]["prefill_workers"] = args.prefill_workers
        metadata["run_metadata"]["decode_workers"] = args.decode_workers

    # Add multiple frontends info if enabled
    if args.enable_multiple_frontends:
        metadata["run_metadata"]["enable_multiple_frontends"] = True
        metadata["run_metadata"]["num_additional_frontends"] = (
            args.num_additional_frontends
        )

    return metadata


def _get_available_gpu_types() -> list[str]:
    """Discover available GPU types by scanning scripts directory structure.

    Looks for scripts in: scripts/{gpu_type}/{agg,disagg}/*.sh
    """
    script_dir = pathlib.Path(__file__).parent / "scripts"
    gpu_types = set()

    # Scan for GPU type directories (directories that contain agg/ or disagg/)
    for gpu_dir in script_dir.iterdir():
        if not gpu_dir.is_dir():
            continue

        # Check if this directory has agg/ or disagg/ subdirectories
        has_agg = (gpu_dir / "agg").is_dir()
        has_disagg = (gpu_dir / "disagg").is_dir()

        if has_agg or has_disagg:
            gpu_types.add(gpu_dir.name)

    return sorted(list(gpu_types))


def _parse_command_line_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM job scripts"
    )

    # Get available GPU types dynamically
    available_gpu_types = _get_available_gpu_types()

    # Template parameters
    parser.add_argument("--job-name", default="dynamo_setup", help="SLURM job name")
    parser.add_argument("--account", default=None, help="SLURM account (or set in srtslurm.yaml)")
    parser.add_argument("--model-dir", required=True, help="Model directory path")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Config directory with dynamo wheels/binaries (default: ../configs from slurm_runner/)",
    )
    parser.add_argument(
        "--container-image",
        default=None,
        help="Container image (or set in srtslurm.yaml)",
    )
    parser.add_argument(
        "--time-limit", default=None, help="Time limit (default: 04:00:00 or from srtslurm.yaml)"
    )
    parser.add_argument(
        "--prefill-nodes", type=int, default=None, help="Number of prefill nodes"
    )
    parser.add_argument(
        "--decode-nodes", type=int, default=None, help="Number of decode nodes"
    )
    parser.add_argument(
        "--prefill-workers", type=int, default=None, help="Number of prefill workers"
    )
    parser.add_argument(
        "--decode-workers", type=int, default=None, help="Number of decode workers"
    )
    parser.add_argument(
        "--agg-nodes", type=int, default=None, help="Number of aggregated worker nodes"
    )
    parser.add_argument(
        "--agg-workers", type=int, default=None, help="Number of aggregated workers"
    )
    parser.add_argument(
        "--gpus-per-node", type=int, default=None, help="Number of GPUs per node (or set in srtslurm.yaml)"
    )
    parser.add_argument(
        "--network-interface", default=None, help="Network interface (or set in srtslurm.yaml)"
    )
    parser.add_argument(
        "--gpu-type",
        choices=available_gpu_types,
        default=available_gpu_types[0] if available_gpu_types else None,
        help=f"GPU type to use. Available types: {', '.join(available_gpu_types)}",
    )
    parser.add_argument(
        "--script-variant",
        type=str,
        required=True,
        help="Script variant to use (e.g., 'max-tpt', '1p_4d'). Corresponds to the .sh filename without extension.",
    )

    parser.add_argument(
        "--partition",
        default=None,
        help="SLURM partition (or set in srtslurm.yaml)",
    )
    parser.add_argument(
        "--enable-multiple-frontends",
        action="store_true",
        default=True,
        help="Enable multiple frontend architecture with nginx load balancer (default: True)",
    )
    parser.add_argument(
        "--disable-multiple-frontends",
        action="store_false",
        dest="enable_multiple_frontends",
        help="Disable multiple frontends (use single frontend)",
    )
    parser.add_argument(
        "--num-additional-frontends",
        type=int,
        default=9,
        help="Number of additional frontend nodes (default: 9)",
    )

    parser.add_argument(
        "--use-init-location",
        action="store_true",
        help="Whether we use '--init-expert-locations' json files",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        help="Benchmark configurations. Example: "
        + '"type=sa-bench; isl=8192; osl=1024; concurrencies=16x2048x4096x8192; req-rate=inf"',
    )

    parser.add_argument(
        "--sglang-torch-profiler",
        action="store_true",
        help="Enable torch profiling mode using sglang.launch_server (mutually exclusive with --benchmark)",
    )

    parser.add_argument(
        "--extra-slurm-args",
        action="append",
        default=[],
        help="Extra slurm arguments, remove the '--' prefix. Example: --extra-slurm-args dependency=afterok:<x>",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Tries to launch the job multiple times to catch transient errors",
    )

    parser.add_argument(
        "--disable-config-dump",
        action="store_false",
        dest="enable_config_dump",
        default=True,
        help="Disable dumping config to file on each node (default: config dump is enabled)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to save logs (default: repo root/logs). Path relative to slurm_runner/ or absolute.",
    )

    return parser.parse_args(args)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate arguments and ensure aggregated and disaggregated args are mutually exclusive."""
    # Validate profiling mode constraints
    if args.sglang_torch_profiler:
        if args.benchmark:
            raise ValueError(
                "Cannot specify both --sglang-torch-profiler and --benchmark. "
                "Profiling mode is mutually exclusive with benchmarking."
            )

        # Check for multiple workers in profiling mode
        if hasattr(args, 'agg_workers') and args.agg_workers and args.agg_workers > 1:
            raise ValueError(
                "Profiling mode requires single worker only. "
                f"Got --agg-workers={args.agg_workers}"
            )
        if hasattr(args, 'prefill_workers') and args.prefill_workers and args.prefill_workers > 1:
            raise ValueError(
                "Profiling mode requires single worker only. "
                f"Got --prefill-workers={args.prefill_workers}"
            )
        if hasattr(args, 'decode_workers') and args.decode_workers and args.decode_workers > 1:
            raise ValueError(
                "Profiling mode requires single worker only. "
                f"Got --decode-workers={args.decode_workers}"
            )

    # Config file is in repo root (parent of slurm_runner/)
    config_path = str(pathlib.Path(__file__).parent.parent / "srtslurm.yaml")

    # Validate cluster settings with config file fallback
    try:
        args.account, args.partition, args.network_interface = validate_cluster_settings(
            args.account, args.partition, args.network_interface, config_path
        )
    except ValueError as e:
        raise ValueError(f"Cluster configuration error: {e}")
    
    # Apply time limit default
    if args.time_limit is None:
        args.time_limit = get_cluster_setting("time_limit", None, config_path) or "04:00:00"

    # Apply gpus_per_node from config or require it
    if args.gpus_per_node is None:
        args.gpus_per_node = get_cluster_setting("gpus_per_node", None, config_path)
        if args.gpus_per_node is None:
            raise ValueError(
                "Missing required setting: --gpus-per-node (or cluster.gpus_per_node in srtslurm.yaml)"
            )

    has_disagg_args = any(
        [
            args.prefill_nodes is not None,
            args.decode_nodes is not None,
            args.prefill_workers is not None,
            args.decode_workers is not None,
        ]
    )
    has_agg_args = any(
        [
            args.agg_nodes is not None,
            args.agg_workers is not None,
        ]
    )

    if has_disagg_args and has_agg_args:
        raise ValueError(
            "Cannot specify both aggregated (--agg-nodes, --agg-workers) and "
            "disaggregated (--prefill-nodes, --decode-nodes, --prefill-workers, --decode-workers) arguments"
        )

    if has_disagg_args:
        # Validate disaggregated args
        if args.prefill_nodes is None or args.decode_nodes is None:
            raise ValueError(
                "Disaggregated mode requires both --prefill-nodes and --decode-nodes"
            )
        if args.prefill_workers is None or args.decode_workers is None:
            raise ValueError(
                "Disaggregated mode requires both --prefill-workers and --decode-workers"
            )
        if args.prefill_nodes % args.prefill_workers != 0:
            raise ValueError(
                f"Prefill nodes ({args.prefill_nodes}) must be divisible by prefill workers ({args.prefill_workers})"
            )
        if args.decode_nodes % args.decode_workers != 0:
            raise ValueError(
                f"Decode nodes ({args.decode_nodes}) must be divisible by decode workers ({args.decode_workers})"
            )
        # Validate GPU script exists for disaggregated mode
        script_dir = pathlib.Path(__file__).parent / "scripts"
        disagg_dir = script_dir / args.gpu_type / "disagg"
        # Use script variant (defaults to "default")
        script_name = f"{args.script_variant}.sh"
        gpu_script = disagg_dir / script_name
        if not gpu_script.exists():
            raise ValueError(
                f"Disaggregated GPU script not found: {gpu_script}. Available GPU types: {', '.join(_get_available_gpu_types())}"
            )

    if has_agg_args:
        # Validate aggregated args
        if args.agg_nodes is None or args.agg_workers is None:
            raise ValueError(
                "Aggregated mode requires both --agg-nodes and --agg-workers"
            )
        if args.agg_nodes % args.agg_workers != 0:
            raise ValueError(
                f"Aggregated nodes ({args.agg_nodes}) must be divisible by aggregated workers ({args.agg_workers})"
            )
        # Validate aggregated GPU script exists
        script_dir = pathlib.Path(__file__).parent / "scripts"
        # Remove any -prefill or -decode suffix if present
        base_gpu_type = args.gpu_type.replace("-prefill", "").replace("-decode", "")
        agg_dir = script_dir / base_gpu_type / "agg"
        # Use script variant (defaults to "default")
        script_name = f"{args.script_variant}.sh"
        agg_gpu_script = agg_dir / script_name
        if not agg_gpu_script.exists():
            raise ValueError(
                f"Aggregated GPU script not found: {agg_gpu_script}. Available GPU types: {', '.join(_get_available_gpu_types())}"
            )

    if not has_disagg_args and not has_agg_args:
        raise ValueError(
            "Must specify either aggregated (--agg-nodes, --agg-workers) or "
            "disaggregated (--prefill-nodes, --decode-nodes, --prefill-workers, --decode-workers) arguments"
        )


def main(input_args: list[str] | None = None):
    setup_logging()
    args = _parse_command_line_args(input_args)
    
    # Apply config-dir default if not provided
    if args.config_dir is None:
        # Default: ../configs from slurm_runner directory
        default_config_dir = pathlib.Path(__file__).parent.parent / "configs"
        args.config_dir = str(default_config_dir)
    
    # Apply container-image from config if not provided
    if args.container_image is None:
        from cluster_config import get_cluster_setting
        # Config file is in repo root (parent of slurm_runner/)
        config_path = str(pathlib.Path(__file__).parent.parent / "srtslurm.yaml")
        args.container_image = get_cluster_setting("container_image", None, config_path)
        if args.container_image is None:
            raise ValueError(
                "Container image must be specified via --container-image or cluster.container_image in srtslurm.yaml"
            )

    # Validate arguments
    _validate_args(args)

    # Determine mode and set defaults
    is_aggregated = args.agg_nodes is not None

    if is_aggregated:
        agg_nodes = args.agg_nodes
        agg_workers = args.agg_workers
        prefill_nodes = 0
        decode_nodes = 0
        prefill_workers = 0
        decode_workers = 0
        total_nodes = agg_nodes
    else:
        prefill_nodes = args.prefill_nodes
        decode_nodes = args.decode_nodes
        prefill_workers = args.prefill_workers
        decode_workers = args.decode_workers
        agg_nodes = 0
        agg_workers = 0
        total_nodes = prefill_nodes + decode_nodes

    # Validation for multiple frontends
    if args.enable_multiple_frontends:
        if args.num_additional_frontends < 0:
            raise ValueError("Number of additional frontends cannot be negative")

    # parse benchmark configs
    benchmark_config = {}
    if args.benchmark:
        for key_val_pair in args.benchmark.split("; "):
            key, val = key_val_pair.split("=")
            benchmark_config[key] = val

    # validate benchmark configs
    if benchmark_config == {} or benchmark_config["type"] == "manual":
        parsable_config = ""
        benchmark_config["type"] = "manual"
    elif benchmark_config["type"] == "sa-bench":
        parsable_config = ""
        need_keys = ["isl", "osl", "concurrencies", "req-rate"]
        assert all([key in benchmark_config for key in need_keys])
        assert benchmark_config["isl"].isnumeric()
        parsable_config = f"{parsable_config} {benchmark_config['isl']}"
        assert benchmark_config["osl"].isnumeric()
        parsable_config = f"{parsable_config} {benchmark_config['osl']}"
        assert all([x.isnumeric() for x in benchmark_config["concurrencies"].split("x")])
        parsable_config = f"{parsable_config} {benchmark_config['concurrencies']}"
        assert (
            benchmark_config["req-rate"] == "inf"
            or benchmark_config["req-rate"].isnumeric()
        )
        parsable_config = f"{parsable_config} {benchmark_config['req-rate']}"
    elif benchmark_config["type"] == "gpqa":
        parsable_config = ""
        # Parse gpqa-specific parameters
        if "num-examples" in benchmark_config:
            assert benchmark_config["num-examples"].isnumeric()
            parsable_config = f"{parsable_config} {benchmark_config['num-examples']}"
        else:
            parsable_config = f"{parsable_config} 198"  # default

        if "max-tokens" in benchmark_config:
            assert benchmark_config["max-tokens"].isnumeric()
            parsable_config = f"{parsable_config} {benchmark_config['max-tokens']}"
        else:
            parsable_config = f"{parsable_config} 512"  # default

        if "repeat" in benchmark_config:
            assert benchmark_config["repeat"].isnumeric()
            parsable_config = f"{parsable_config} {benchmark_config['repeat']}"
        else:
            parsable_config = f"{parsable_config} 8"  # default

        if "num-threads" in benchmark_config:
            assert benchmark_config["num-threads"].isnumeric()
            parsable_config = f"{parsable_config} {benchmark_config['num-threads']}"
        else:
            parsable_config = f"{parsable_config} 512"  # default

        if "thinking-mode" in benchmark_config:
            assert benchmark_config["thinking-mode"] in ["deepseek-r1", "deepseek-v3"]
            parsable_config = f"{parsable_config} {benchmark_config['thinking-mode']}"
        else:
            parsable_config = f"{parsable_config} deepseek-r1"  # default
    elif benchmark_config["type"] == "mmlu":
        parsable_config = ""
        # Parse mmlu-specific parameters
        if "num-examples" in benchmark_config:
            assert benchmark_config["num-examples"].isnumeric()
            parsable_config = f"{parsable_config} {benchmark_config['num-examples']}"
        else:
            parsable_config = f"{parsable_config} 200"  # default

        if "max-tokens" in benchmark_config:
            assert benchmark_config["max-tokens"].isnumeric()
            parsable_config = f"{parsable_config} {benchmark_config['max-tokens']}"
        else:
            parsable_config = f"{parsable_config} 128"  # default

        if "repeat" in benchmark_config:
            assert benchmark_config["repeat"].isnumeric()
            parsable_config = f"{parsable_config} {benchmark_config['repeat']}"
        else:
            parsable_config = f"{parsable_config} 1"  # default

        if "num-threads" in benchmark_config:
            assert benchmark_config["num-threads"].isnumeric()
            parsable_config = f"{parsable_config} {benchmark_config['num-threads']}"
        else:
            parsable_config = f"{parsable_config} 128"  # default
    else:
        assert False, benchmark_config["type"]

    # Generate timestamp for log directory naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine base log directory (default: repo root)
    # This is used by the template to set SBATCH output/error paths
    if args.log_dir:
        base_log_dir = pathlib.Path(args.log_dir)
        if not base_log_dir.is_absolute():
            # Relative to slurm_runner directory - use as-is in template
            log_dir_prefix = args.log_dir
        else:
            # Absolute path - compute relative from slurm_runner/
            slurm_runner_dir = pathlib.Path(__file__).parent
            try:
                log_dir_prefix = str(base_log_dir.relative_to(slurm_runner_dir))
            except ValueError:
                # Not relative to slurm_runner, use absolute
                log_dir_prefix = str(base_log_dir)
    else:
        # Default: repo root/logs (parent of slurm_runner/)/logs = "../logs" from slurm_runner/
        log_dir_prefix = "../logs"

    # Select template based on mode
    if is_aggregated:
        template_path = "job_script_template_agg.j2"
    else:
        template_path = "job_script_template_disagg.j2"

    template_vars = {
        "job_name": args.job_name,
        "total_nodes": total_nodes,
        "account": args.account,
        "time_limit": args.time_limit,
        "prefill_nodes": prefill_nodes,
        "decode_nodes": decode_nodes,
        "prefill_workers": prefill_workers,
        "decode_workers": decode_workers,
        "agg_nodes": agg_nodes,
        "agg_workers": agg_workers,
        "is_aggregated": is_aggregated,
        "model_dir": args.model_dir,
        "config_dir": args.config_dir,
        "container_image": args.container_image,
        "gpus_per_node": args.gpus_per_node,
        "network_interface": args.network_interface,
        "gpu_type": args.gpu_type,
        "script_variant": args.script_variant,
        "partition": args.partition,
        "enable_multiple_frontends": args.enable_multiple_frontends,
        "num_additional_frontends": args.num_additional_frontends,
        "use_init_location": args.use_init_location,
        "do_benchmark": benchmark_config["type"] != "manual" if not args.sglang_torch_profiler else False,
        "benchmark_type": benchmark_config["type"],
        "benchmark_arg": parsable_config,
        "timestamp": timestamp,
        "enable_config_dump": args.enable_config_dump,
        "use_dynamo_whls": True,  # Always true when config-dir is set
        "log_dir_prefix": log_dir_prefix,
        "profiling_enabled": args.sglang_torch_profiler,
        "profiling_dir": "/logs/profiles",
        "profiling_mode": "decode",  # Will be overridden for prefill job in disagg mode
    }

    # Create temporary file for sbatch script
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        submitted_job_ids = []

        # Handle disaggregated profiling mode: submit 2 separate jobs (prefill + decode)
        # Use aggregated template for each job since they're standalone
        if args.sglang_torch_profiler and not is_aggregated:
            # Submit prefill profiling job (use aggregated template)
            template_vars_prefill = template_vars.copy()
            template_vars_prefill["profiling_mode"] = "prefill"
            template_vars_prefill["total_nodes"] = prefill_nodes
            template_vars_prefill["agg_nodes"] = prefill_nodes
            template_vars_prefill["agg_workers"] = prefill_workers
            template_vars_prefill["is_aggregated"] = True

            _, rendered_script_prefill = generate_job_script(
                "job_script_template_agg.j2", temp_path, **template_vars_prefill
            )

            prefill_job_id = submit_job(temp_path, args.extra_slurm_args)
            submitted_job_ids.append(prefill_job_id)
            logging.info(f"Submitted prefill profiling job: {prefill_job_id}")

            # Submit decode profiling job (use aggregated template)
            template_vars_decode = template_vars.copy()
            template_vars_decode["profiling_mode"] = "decode"
            template_vars_decode["total_nodes"] = decode_nodes
            template_vars_decode["agg_nodes"] = decode_nodes
            template_vars_decode["agg_workers"] = decode_workers
            template_vars_decode["is_aggregated"] = True

            _, rendered_script_decode = generate_job_script(
                "job_script_template_agg.j2", temp_path, **template_vars_decode
            )

            decode_job_id = submit_job(temp_path, args.extra_slurm_args)
            submitted_job_ids.append(decode_job_id)
            logging.info(f"Submitted decode profiling job: {decode_job_id}")

            # Store both scripts for later saving
            rendered_script = f"# Prefill Job:\n{rendered_script_prefill}\n\n# Decode Job:\n{rendered_script_decode}"
            job_id = prefill_job_id  # Use prefill job ID for log directory naming
        else:
            # Normal mode or aggregated profiling mode: single job submission
            _, rendered_script = generate_job_script(
                template_path, temp_path, **template_vars
            )

            job_id = submit_job(temp_path, args.extra_slurm_args)
            submitted_job_ids.append(job_id)

        # Create log directory with new naming format IMMEDIATELY after submission
        # SLURM will write log.out/log.err to this directory when job starts
        if is_aggregated:
            log_dir_name = f"{job_id}_{agg_workers}A_{timestamp}"
        else:
            log_dir_name = f"{job_id}_{prefill_workers}P_{decode_workers}D_{timestamp}"
        
        # Determine base log directory (default: repo root/logs)
        if args.log_dir:
            base_log_dir = pathlib.Path(args.log_dir)
            if not base_log_dir.is_absolute():
                # Relative to slurm_runner directory
                base_log_dir = pathlib.Path(__file__).parent / base_log_dir
        else:
            # Default: repo root/logs (parent directory of slurm_runner/ + logs)
            base_log_dir = pathlib.Path(__file__).parent.parent / "logs"
        
        log_dir_path = base_log_dir / log_dir_name
        os.makedirs(log_dir_path, exist_ok=True)

        # Save rendered sbatch script
        sbatch_script_path = os.path.join(log_dir_path, "sbatch_script.sh")
        with open(sbatch_script_path, "w") as f:
            f.write(rendered_script)
        logging.info(f"Saved rendered sbatch script to {sbatch_script_path}")

        # Create and save job metadata
        metadata = create_job_metadata(
            job_id=job_id,
            timestamp=timestamp,
            args=args,
            benchmark_config=benchmark_config,
        )
        metadata_path = os.path.join(log_dir_path, f"{job_id}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved job metadata to {metadata_path}")

        # retries logic
        if args.retries > 0:
            extra_slurm_args_without_dependencies = [
                x for x in args.extra_slurm_args if "dependency" not in x
            ]
            for _ in range(args.retries):
                dependencies = ",".join(
                    [f"afternotok:{job}" for job in submitted_job_ids]
                )
                slurm_args = extra_slurm_args_without_dependencies + [
                    f"dependency={dependencies}"
                ]
                job_id = submit_job(temp_path, slurm_args)
                submitted_job_ids.append(job_id)

                # Save script for retry job as well
                if is_aggregated:
                    retry_log_dir_name = f"{job_id}_{agg_workers}A_{timestamp}"
                else:
                    retry_log_dir_name = (
                        f"{job_id}_{prefill_workers}P_{decode_workers}D_{timestamp}"
                    )
                retry_log_dir_path = base_log_dir / retry_log_dir_name
                os.makedirs(retry_log_dir_path, exist_ok=True)
                retry_sbatch_script_path = os.path.join(
                    retry_log_dir_path, "sbatch_script.sh"
                )
                with open(retry_sbatch_script_path, "w") as f:
                    f.write(rendered_script)
                logging.info(
                    f"Saved rendered sbatch script to {retry_sbatch_script_path}"
                )

                # Create and save job metadata for retry job
                retry_metadata = create_job_metadata(
                    job_id=job_id,
                    timestamp=timestamp,
                    args=args,
                    benchmark_config=benchmark_config,
                )
                retry_metadata_path = os.path.join(retry_log_dir_path, f"{job_id}.json")
                with open(retry_metadata_path, "w") as f:
                    json.dump(retry_metadata, f, indent=2)
                logging.info(f"Saved job metadata to {retry_metadata_path}")

        print_welcome_message(submitted_job_ids, log_dir_name)
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
