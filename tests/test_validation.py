#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for config validation (resource constraints, profiling mode, etc).
"""

import pytest
from pydantic import ValidationError
from srtctl.core.schema import JobConfig


def test_valid_disaggregated_config():
    """Test that valid disaggregated config passes validation."""
    config = {
        "name": "test-valid-disagg",
        "model": {
            "path": "/models/test",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "prefill_nodes": 1,
            "decode_nodes": 4,
            "prefill_workers": 1,
            "decode_workers": 4,
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test",
            "partition": "test"
        },
        "backend": {
            "sglang_config": {
                "prefill": {
                    "tensor-parallel-size": 4
                },
                "decode": {
                    "tensor-parallel-size": 4
                }
            }
        }
    }

    # Should not raise
    validated = JobConfig(**config)
    assert validated.name == "test-valid-disagg"


def test_invalid_tp_size_too_large():
    """Test that TP size larger than available GPUs fails validation."""
    config = {
        "name": "test-invalid-tp",
        "model": {
            "path": "/models/test",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "prefill_nodes": 1,  # 1 node × 4 GPUs = 4 total GPUs
            "decode_nodes": 4,
            "prefill_workers": 1,
            "decode_workers": 4,
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test",
            "partition": "test"
        },
        "backend": {
            "sglang_config": {
                "prefill": {
                    "tensor-parallel-size": 8  # ERROR: Need 8 GPUs but only have 4!
                },
                "decode": {
                    "tensor-parallel-size": 4
                }
            }
        }
    }

    with pytest.raises(ValueError, match="Prefill resource mismatch"):
        JobConfig(**config)


def test_invalid_too_many_workers():
    """Test that too many workers for available GPUs fails validation."""
    config = {
        "name": "test-too-many-workers",
        "model": {
            "path": "/models/test",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "prefill_nodes": 1,  # 1 node × 4 GPUs = 4 total GPUs
            "decode_nodes": 4,
            "prefill_workers": 2,  # 2 workers × 4 GPUs = 8 needed, but only have 4!
            "decode_workers": 4,
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test",
            "partition": "test"
        },
        "backend": {
            "sglang_config": {
                "prefill": {
                    "tensor-parallel-size": 4
                },
                "decode": {
                    "tensor-parallel-size": 4
                }
            }
        }
    }

    with pytest.raises(ValidationError, match="Prefill resource mismatch"):
        JobConfig(**config)


def test_invalid_profiling_with_multiple_workers():
    """Test that profiling mode with multiple workers fails validation."""
    config = {
        "name": "test-profiling-multi-worker",
        "model": {
            "path": "/models/test",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "prefill_nodes": 1,
            "decode_nodes": 4,
            "prefill_workers": 2,  # ERROR: Profiling requires single worker
            "decode_workers": 1,
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test",
            "partition": "test"
        },
        "backend": {
            "enable_profiling": True,
            "sglang_config": {
                "prefill": {
                    "tensor-parallel-size": 4
                },
                "decode": {
                    "tensor-parallel-size": 4
                }
            }
        }
    }

    with pytest.raises(ValueError, match="Profiling mode requires single worker only.*prefill_workers=2"):
        JobConfig(**config)


def test_invalid_profiling_with_benchmark():
    """Test that profiling mode with benchmarking fails validation."""
    config = {
        "name": "test-profiling-benchmark",
        "model": {
            "path": "/models/test",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "prefill_nodes": 1,
            "decode_nodes": 4,
            "prefill_workers": 1,
            "decode_workers": 1,
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test",
            "partition": "test"
        },
        "backend": {
            "enable_profiling": True,
            "sglang_config": {
                "prefill": {"tensor-parallel-size": 4},
                "decode": {"tensor-parallel-size": 4}
            }
        },
        "benchmark": {
            "type": "sa-bench",  # ERROR: Can't benchmark while profiling
            "isl": 1024,
            "osl": 1024
        }
    }

    with pytest.raises(ValidationError, match="Cannot enable profiling with benchmark type"):
        JobConfig(**config)


def test_valid_aggregated_config():
    """Test that valid aggregated config passes validation."""
    config = {
        "name": "test-valid-agg",
        "model": {
            "path": "/models/test",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "agg_nodes": 4,
            "agg_workers": 4,
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test",
            "partition": "test"
        },
        "backend": {
            "sglang_config": {
                "aggregated": {
                    "tensor-parallel-size": 4
                }
            }
        }
    }

    # Should not raise
    validated = JobConfig(**config)
    assert validated.name == "test-valid-agg"


def test_invalid_aggregated_profiling_multi_worker():
    """Test that aggregated profiling with multiple workers fails."""
    config = {
        "name": "test-agg-profiling-multi",
        "model": {
            "path": "/models/test",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "agg_nodes": 4,
            "agg_workers": 4,  # ERROR: Profiling requires single worker
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test",
            "partition": "test"
        },
        "backend": {
            "enable_profiling": True,
            "sglang_config": {
                "aggregated": {
                    "tensor-parallel-size": 4
                }
            }
        }
    }

    with pytest.raises(ValueError, match="Profiling mode requires single worker only.*agg_workers=4"):
        JobConfig(**config)


def test_valid_multi_node_tp():
    """Test that multi-node TP configuration is valid."""
    config = {
        "name": "test-multi-node-tp",
        "model": {
            "path": "/models/test",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "prefill_nodes": 2,  # 2 nodes × 4 GPUs = 8 GPUs
            "decode_nodes": 4,
            "prefill_workers": 1,  # 1 worker using 8 GPUs (TP=8)
            "decode_workers": 4,
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test",
            "partition": "test"
        },
        "backend": {
            "sglang_config": {
                "prefill": {
                    "tensor-parallel-size": 8  # Valid: 1 worker × 8 GPUs = 8 total, matches 2 nodes × 4 GPUs
                },
                "decode": {
                    "tensor-parallel-size": 4
                }
            }
        }
    }

    # Should not raise
    validated = JobConfig(**config)
    assert validated.name == "test-multi-node-tp"


def test_template_placeholder_skips_validation():
    """Test that template placeholders like {tp_size} skip validation."""
    config = {
        "name": "test-template",
        "model": {
            "path": "/models/test",
            "container": "test.sqsh",
            "precision": "fp8"
        },
        "resources": {
            "gpu_type": "gb200",
            "prefill_nodes": 1,
            "decode_nodes": 4,
            "prefill_workers": 1,
            "decode_workers": 4,
            "gpus_per_node": 4
        },
        "slurm": {
            "account": "test",
            "partition": "test"
        },
        "backend": {
            "sglang_config": {
                "prefill": {
                    "tensor-parallel-size": "{tp_size}"  # Template placeholder - should skip validation
                },
                "decode": {
                    "tensor-parallel-size": 4
                }
            }
        }
    }

    # Should not raise (template placeholders are skipped)
    validated = JobConfig(**config)
    assert validated.name == "test-template"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
