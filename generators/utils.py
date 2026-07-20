"""
Utility validation functions for synthetic data generators.

This module contains helper functions used throughout the synthetic
dataset generators to validate user inputs and ensure consistent
behavior across the library.

Author
------
Sarah Elyane Rashiwa
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np


def validate_function(
    function: str,
    available_functions: Tuple[str, ...],
) -> None:
    """
    Validate that the requested function is supported.

    Parameters
    ----------
    function : str
        Name of the regression function.

    available_functions : tuple of str
        Tuple containing the supported function names.

    Raises
    ------
    ValueError
        If the requested function is not supported.
    """
    function = function.lower().strip()

    if function not in available_functions:
        raise ValueError(
            f"Unsupported function '{function}'. "
            f"Available functions are: "
            f"{', '.join(sorted(available_functions))}."
        )


def validate_noise(noise: float) -> None:
    """
    Validate the noise level.

    Parameters
    ----------
    noise : float
        Standard deviation of the Gaussian noise.

    Raises
    ------
    ValueError
        If the noise level is negative.
    """
    if noise < 0:
        raise ValueError("Noise must be non-negative.")


def validate_proportion(
    proportion: Tuple[float, ...],
) -> None:
    """
    Validate the proportions used in a multimodal generator.

    Parameters
    ----------
    proportion : tuple of float
        Relative proportions assigned to each mode.

    Raises
    ------
    ValueError
        If the proportions are invalid.
    """
    if not isinstance(proportion, tuple):
        raise ValueError("Proportion must be provided as a tuple.")

    if len(proportion) < 2:
        raise ValueError(
            "At least two proportions are required."
        )

    if not all(isinstance(p, (int, float)) for p in proportion):
        raise ValueError(
            "All proportions must be numeric."
        )

    if any(p < 0 for p in proportion):
        raise ValueError(
            "Proportions must be non-negative."
        )

    if not np.isclose(sum(proportion), 1.0):
        raise ValueError(
            "Proportions must sum to 1."
        )


def validate_breakpoints(
    breakpoints: Tuple[float, ...],
) -> None:
    """
    Validate the breakpoints used in a piecewise generator.

    Parameters
    ----------
    breakpoints : tuple of float
        Ordered breakpoints defining the piecewise regions.

    Raises
    ------
    ValueError
        If the breakpoints are invalid.
    """
    if not isinstance(breakpoints, tuple):
        raise ValueError(
            "Breakpoints must be provided as a tuple."
        )

    if len(breakpoints) == 0:
        raise ValueError(
            "At least one breakpoint must be provided."
        )

    if not all(isinstance(b, (int, float)) for b in breakpoints):
        raise ValueError(
            "All breakpoints must be numeric."
        )

    if any(b <= 0 or b >= 1 for b in breakpoints):
        raise ValueError(
            "Breakpoints must lie strictly between 0 and 1."
        )

    if any(b1 >= b2 for b1, b2 in zip(breakpoints, breakpoints[1:])):
        raise ValueError(
            "Breakpoints must be strictly increasing."
        )
        
def check_random_state(random_state: Optional[int] = None) -> np.random.RandomState:
    """
    Return a NumPy RandomState instance.

    Parameters
    ----------
    random_state : None, int or RandomState

    Returns
    -------
    RandomState
    """
    if random_state is None:
        return np.random.RandomState()

    if isinstance(random_state, int):
        return np.random.RandomState(random_state)

    if isinstance(random_state, RandomState):
        return random_state

    raise TypeError(
        "random_state must be None, int or numpy.random.RandomState."
    )