#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SGLang backend support.
"""

import logging
import os
import tempfile
import yaml
from datetime import datetime
from jinja2 import Template
from pathlib import Path
from typing import Any

from .base import Backend


def expand_template(template: Any, values: dict[str, Any]) -> Any:
    """Recursively expand template strings with values.
    
    Used for parameter sweeping - replaces {param_name} with actual values.
    """
    if isinstance(template, dict):
        return {k: expand_template(v, values) for k, v in template.items()}
    elif isinstance(template, list):
        return [expand_template(item, values) for item in template]
    elif isinstance(template, str):
        result = template
        for key, value in values.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        return result
    else:
        return template


class SGLangBackend(Backend):
    """SGLang backend for distributed serving."""
    
    def generate_config_file(self, params: dict = None) -> Path | None:
        """Generate SGLang YAML config file.
        
        Args:
            params: Optional sweep parameters for template expansion
            
        Returns:
            Path to generated config file
        """
        if 'sglang_config' not in self.backend_config:
            return None

        sglang_cfg = self.backend_config['sglang_config']

        # Expand templates if sweeping
        if params:
            sglang_cfg = expand_template(sglang_cfg, params)
            logging.info(f"Expanded config with params: {params}")

        # Extract prefill and decode configs (no merging)
        result = {}
        for mode in ['prefill', 'decode']:
            if mode in sglang_cfg:
                result[mode] = sglang_cfg[mode]

        # Write to temp file
        fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='sglang_config_')
        with os.fdopen(fd, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)

        logging.info(f"Generated SGLang config: {temp_path}")
        return Path(temp_path)
    
    def render_command(self, mode: str, config_path: Path = None) -> str:
        """Render full SGLang command with all flags inlined.
        
        Args:
            mode: "prefill" or "decode"
            config_path: Path to generated SGLang config file
            
        Returns:
            Multi-line bash command string
        """
        lines = []
        
        # Environment variables
        env_vars = self.get_environment_vars(mode)
        for key, val in env_vars.items():
            lines.append(f"{key}={val} \\")
        
        # Python command
        lines.append("python3 -m dynamo.sglang \\")
        
        # Inline all SGLang flags from config file
        if config_path:
            with open(config_path) as f:
                sglang_config = yaml.load(f, Loader=yaml.FullLoader)
            
            mode_config = sglang_config.get(mode, {})
            flag_lines = self._config_to_flags(mode_config)
            lines.extend(flag_lines)
        
        # Add coordination flags
        coord_flags = self._get_coordination_flags(mode)
        lines.extend(coord_flags)
        
        return "\n".join(lines)
    
    def _config_to_flags(self, config: dict) -> list[str]:
        """Convert config dict to CLI flags.
        
        Args:
            config: SGLang config dict for this mode
            
        Returns:
            List of flag strings with backslash continuations
        """
        lines = []
        
        for key, value in sorted(config.items()):
            # Convert underscores to hyphens
            flag_name = key.replace('_', '-')
            
            if isinstance(value, bool):
                if value:
                    lines.append(f"    --{flag_name} \\")
            elif isinstance(value, list):
                values_str = " ".join(str(v) for v in value)
                lines.append(f"    --{flag_name} {values_str} \\")
            else:
                lines.append(f"    --{flag_name} {value} \\")
        
        return lines
    
    def _get_coordination_flags(self, mode: str) -> list[str]:
        """Get multi-node coordination flags.
        
        Args:
            mode: "prefill" or "decode"
            
        Returns:
            List of coordination flag strings
        """
        lines = []
        
        # Determine nnodes based on mode
        if self.is_disaggregated():
            nnodes = (self.resources['prefill_nodes'] 
                     if mode == 'prefill' 
                     else self.resources['decode_nodes'])
        else:
            nnodes = self.resources['agg_nodes']
        
        # Coordination flags
        lines.append("    --dist-init-addr $HOST_IP_MACHINE:$PORT \\")
        lines.append(f"    --nnodes {nnodes} \\")
        lines.append("    --node-rank $RANK \\")
        
        # Parallelism flags
        gpus_per_node = self.resources.get('gpus_per_node', 4)
        lines.append(f"    --ep-size {gpus_per_node} \\")
        lines.append(f"    --tp-size {gpus_per_node} \\")
        lines.append(f"    --dp-size {gpus_per_node}")

        return lines

    def generate_slurm_script(self, config_path: Path = None, timestamp: str = None) -> tuple[Path, str]:
        """Generate SLURM job script from Jinja template.

        Args:
            config_path: Path to SGLang config file
            timestamp: Timestamp for job submission

        Returns:
            Tuple of (script_path, rendered_script_content)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine mode and node counts
        is_aggregated = not self.is_disaggregated()

        if is_aggregated:
            agg_nodes = self.resources['agg_nodes']
            agg_workers = self.resources['agg_workers']
            prefill_nodes = 0
            decode_nodes = 0
            prefill_workers = 0
            decode_workers = 0
            total_nodes = agg_nodes
        else:
            prefill_nodes = self.resources['prefill_nodes']
            decode_nodes = self.resources['decode_nodes']
            prefill_workers = self.resources['prefill_workers']
            decode_workers = self.resources['decode_workers']
            agg_nodes = 0
            agg_workers = 0
            total_nodes = prefill_nodes + decode_nodes

        # Get SLURM settings
        job_name = self.config.get('name', 'infbench-job')
        account = self.slurm.get('account')
        partition = self.slurm.get('partition')
        time_limit = self.slurm.get('time_limit', '01:00:00')

        # Get resource settings from srtslurm.yaml if available
        from infbench.core.config import get_srtslurm_setting
        gpus_per_node = get_srtslurm_setting('gpus_per_node', self.resources.get('gpus_per_node'))
        network_interface = get_srtslurm_setting('network_interface', None)

        # Get backend settings
        gpu_type = self.backend_config.get('gpu_type', 'h100')
        script_variant = self.backend_config.get('script_variant', 'default')

        # Benchmark config
        benchmark_config = self.config.get('benchmark', {})
        bench_type = benchmark_config.get('type', 'manual')
        do_benchmark = bench_type != 'manual'

        # Parse benchmark args if applicable
        parsable_config = ""
        if bench_type == 'sa-bench':
            isl = benchmark_config.get('isl')
            osl = benchmark_config.get('osl')
            concurrencies = benchmark_config.get('concurrencies')
            req_rate = benchmark_config.get('req_rate', 'inf')

            if isinstance(concurrencies, list):
                concurrency_str = "x".join(str(c) for c in concurrencies)
            else:
                concurrency_str = str(concurrencies)

            parsable_config = f"{isl} {osl} {concurrency_str} {req_rate}"

        # Config directory should point to where deepep_config.json lives
        # This is typically the configs/ directory in the yaml-config repo
        import infbench
        yaml_config_root = Path(infbench.__file__).parent.parent.parent
        config_dir_path = yaml_config_root / "configs"

        # Log directory - relative path from slurm_runner/ to infbench/logs
        # Template will be run from slurm_runner/, so ../logs points to infbench/logs
        infbench_root = yaml_config_root.parent / "infbench"
        log_dir_path = infbench_root / "logs"

        # Template variables
        template_vars = {
            "job_name": job_name,
            "total_nodes": total_nodes,
            "account": account,
            "time_limit": time_limit,
            "prefill_nodes": prefill_nodes,
            "decode_nodes": decode_nodes,
            "prefill_workers": prefill_workers,
            "decode_workers": decode_workers,
            "agg_nodes": agg_nodes,
            "agg_workers": agg_workers,
            "is_aggregated": is_aggregated,
            "model_dir": self.model.get('path'),
            "config_dir": str(config_dir_path),
            "container_image": self.model.get('container'),
            "gpus_per_node": gpus_per_node,
            "network_interface": network_interface,
            "gpu_type": gpu_type,
            "script_variant": script_variant,
            "partition": partition,
            "enable_multiple_frontends": self.backend_config.get('enable_multiple_frontends', True),
            "num_additional_frontends": self.backend_config.get('num_additional_frontends', 9),
            "use_init_location": self.config.get('use_init_location', False),
            "do_benchmark": do_benchmark,
            "benchmark_type": bench_type,
            "benchmark_arg": parsable_config,
            "timestamp": timestamp,
            "enable_config_dump": self.config.get('enable_config_dump', True),
            "use_dynamo_whls": True,
            "log_dir_prefix": "../logs",  # Relative to slurm_runner/
            "sglang_torch_profiler": False,
        }

        # Select template based on mode
        if is_aggregated:
            template_name = "job_script_template_agg.j2"
        else:
            template_name = "job_script_template_disagg.j2"

        # Find template path - templates are in ../infbench/slurm_runner
        # relative to the infbench-yaml-config directory
        import infbench
        yaml_config_root = Path(infbench.__file__).parent.parent.parent
        template_path = yaml_config_root.parent / "infbench" / "slurm_runner" / template_name

        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}\n"
                f"Expected template at: {template_path}\n"
                f"Make sure infbench repo with slurm_runner/ is at: {yaml_config_root.parent / 'infbench'}"
            )

        # Render template
        with open(template_path) as f:
            template = Template(f.read())

        rendered_script = template.render(**template_vars)

        # Write to temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.sh', prefix='slurm_job_')
        with os.fdopen(fd, 'w') as f:
            f.write(rendered_script)

        logging.info(f"Generated SLURM job script: {temp_path}")
        return Path(temp_path), rendered_script
