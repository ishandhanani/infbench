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
