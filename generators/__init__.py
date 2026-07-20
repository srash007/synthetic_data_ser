"""
Synthetic data generators for imbalanced regression.

This package provides a collection of generators used to create
synthetic regression datasets under different controlled conditions.
"""

from .base import SyntheticGenerator
from .function_generator import FunctionGenerator
from .multimodal import MultiModalGenerator
from .utils import (
    validate_function,
    check_random_state,
)

__all__ = [
    "SyntheticGenerator",
    "FunctionGenerator",
    "MultiModalGenerator",
    "validate_function",
    "check_random_state",
]