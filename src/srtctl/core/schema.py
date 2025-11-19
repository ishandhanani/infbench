#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic schema definitions for job configuration validation.
"""

from enum import Enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class GpuType(str, Enum):
    """Supported GPU types."""
    GB200 = "gb200"
    H100 = "h100"


class Precision(str, Enum):
    """Model precision/quantization formats."""
    FP4 = "fp4"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"


class BenchmarkType(str, Enum):
    """Benchmark types."""
    MANUAL = "manual"
    SA_BENCH = "sa-bench"
    MMLU = "mmlu"
    GPQA = "gpqa"


class ModelConfig(BaseModel):
    """Model configuration."""
    model_config = {"use_enum_values": True}

    path: str = Field(..., description="Path or alias to model directory")
    container: str = Field(..., description="Path or alias to container image")
    precision: Precision = Field(..., description="Model precision (fp4, fp8, fp16, bf16)")


class ResourceConfig(BaseModel):
    """Resource allocation configuration."""
    model_config = {"use_enum_values": True}

    gpu_type: GpuType = Field(..., description="GPU type (gb200, h100)")
    gpus_per_node: int = Field(4, description="Number of GPUs per node")

    # Disaggregated mode
    prefill_nodes: Optional[int] = Field(None, description="Number of prefill nodes")
    decode_nodes: Optional[int] = Field(None, description="Number of decode nodes")
    prefill_workers: Optional[int] = Field(None, description="Number of prefill workers")
    decode_workers: Optional[int] = Field(None, description="Number of decode workers")

    # Aggregated mode
    agg_nodes: Optional[int] = Field(None, description="Number of aggregated nodes")
    agg_workers: Optional[int] = Field(None, description="Number of aggregated workers")

    @field_validator("prefill_nodes", "decode_nodes", "agg_nodes")
    @classmethod
    def validate_mode(cls, v, info):
        """Validate that either disagg or agg mode is specified."""
        data = info.data
        has_disagg = any(k in data for k in ["prefill_nodes", "decode_nodes"])
        has_agg = "agg_nodes" in data

        if has_disagg and has_agg:
            raise ValueError("Cannot specify both disaggregated and aggregated mode")

        return v


class SlurmConfig(BaseModel):
    """SLURM job settings."""
    account: str = Field(..., description="SLURM account")
    partition: str = Field(..., description="SLURM partition")
    time_limit: str = Field("04:00:00", description="Job time limit (HH:MM:SS)")


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""
    type: BenchmarkType = Field(BenchmarkType.MANUAL, description="Benchmark type")

    # SA-bench specific
    isl: Optional[int] = Field(None, description="Input sequence length")
    osl: Optional[int] = Field(None, description="Output sequence length")
    concurrencies: Optional[list[int]] = Field(None, description="Concurrency levels to test")
    req_rate: Optional[str] = Field("inf", description="Request rate")


class SGLangPrefillConfig(BaseModel):
    """SGLang prefill worker configuration."""
    # Allow any additional fields for SGLang flags
    model_config = {"extra": "allow"}

    # Common required fields
    served_model_name: Optional[str] = None
    model_path: str = "/model/"
    trust_remote_code: bool = True
    disaggregation_mode: Literal["prefill"] = "prefill"


class SGLangDecodeConfig(BaseModel):
    """SGLang decode worker configuration."""
    # Allow any additional fields for SGLang flags
    model_config = {"extra": "allow"}

    # Common required fields
    served_model_name: Optional[str] = None
    model_path: str = "/model/"
    trust_remote_code: bool = True
    disaggregation_mode: Literal["decode"] = "decode"


class SGLangConfig(BaseModel):
    """SGLang backend configuration."""
    prefill: Optional[SGLangPrefillConfig] = None
    decode: Optional[SGLangDecodeConfig] = None


class BackendConfig(BaseModel):
    """Backend configuration (auto-populated, not user-facing)."""
    type: Literal["sglang"] = "sglang"  # Only SGLang supported for now

    # Auto-populated from resources.gpu_type + model.precision
    gpu_type: Optional[str] = None

    # Environment variables
    prefill_environment: Optional[dict[str, str]] = None
    decode_environment: Optional[dict[str, str]] = None

    # SGLang-specific config
    sglang_config: Optional[SGLangConfig] = None

    # Frontend settings
    enable_multiple_frontends: bool = True
    num_additional_frontends: int = 9

    # Profiling settings
    enable_profiling: bool = Field(
        False,
        description="Enable torch profiling mode (uses sglang.launch_server instead of dynamo.sglang)",
    )


class JobConfig(BaseModel):
    """Complete job configuration."""
    model_config = {"use_enum_values": True}

    name: str = Field(..., description="Job name")
    model: ModelConfig
    resources: ResourceConfig
    slurm: SlurmConfig
    backend: Optional[BackendConfig] = None  # Auto-populated
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    # Additional optional settings
    use_init_location: bool = False
    enable_config_dump: bool = True

    def model_post_init(self, __context: Any) -> None:
        """Auto-populate backend config if not provided."""
        if self.backend is None:
            self.backend = BackendConfig()

        # Auto-populate gpu_type from resources (values are already strings due to use_enum_values)
        if self.backend.gpu_type is None:
            self.backend.gpu_type = f"{self.resources.gpu_type}-{self.model.precision}"
