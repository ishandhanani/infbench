"""
InfBench - Benchmark submission framework for distributed serving workloads.
"""

__version__ = "0.1.0"

from .core.config import load_config
from .backends.base import Backend
from .backends.sglang import SGLangBackend

__all__ = [
    "load_config",
    "Backend",
    "SGLangBackend",
]
