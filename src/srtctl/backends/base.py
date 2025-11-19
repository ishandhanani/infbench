#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base backend interface for inference frameworks.

Defines a protocol for framework-specific implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class Backend(ABC):
    """Base class for inference backend implementations.

    Each backend is responsible for:
    1. Generating backend-specific config files
    2. Rendering commands with proper flags and environment variables
    3. Generating SLURM job scripts from Jinja templates
    """

    def __init__(self, config: dict):
        """Initialize backend with user config.

        Args:
            config: Full user configuration dict
        """
        self.config = config
        self.backend_config = config.get("backend", {})
        self.resources = config.get("resources", {})
        self.model = config.get("model", {})
        self.slurm = config.get("slurm", {})

    @abstractmethod
    def generate_config_file(self, params: dict = None) -> Path | None:
        """Generate backend-specific config file.

        Args:
            params: Optional sweep parameters for template expansion

        Returns:
            Path to generated config file, or None if not applicable
        """
        pass

    @abstractmethod
    def render_command(self, mode: str, config_path: Path = None) -> str:
        """Render full command that would be executed.

        Args:
            mode: Worker mode (e.g., "prefill", "decode", "aggregated")
            config_path: Path to generated config file (if applicable)

        Returns:
            Multi-line bash command string with env vars and flags
        """
        pass

    @abstractmethod
    def generate_slurm_script(self, config_path: Path = None, timestamp: str = None) -> tuple[Path, str]:
        """Generate SLURM job script from Jinja template.

        Args:
            config_path: Path to backend-specific config file (if applicable)
            timestamp: Timestamp for job submission (used in log directory naming)

        Returns:
            Tuple of (script_path, rendered_script_content)
        """
        pass

    def get_environment_vars(self, mode: str) -> dict[str, str]:
        """Get environment variables for this mode.

        Args:
            mode: Worker mode

        Returns:
            Dict of environment variable key-value pairs
        """
        env_key = f"{mode}_environment"
        return self.backend_config.get(env_key, {})

    def is_disaggregated(self) -> bool:
        """Check if running in disaggregated mode (has prefill/decode nodes)."""
        return self.resources.get("prefill_nodes") is not None
