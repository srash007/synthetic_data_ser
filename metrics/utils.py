"""
Utility functions used throughout the metrics package.

This module provides helper functions shared by several metrics,
including tail selection and relevance computation.

Author
------
Sarah Rashiwa
"""

import numpy as np


def tail_mask(y, lower=None, upper=None):
    """
    Return a boolean mask identifying observations belonging to
    the lower and/or upper tails of the target distribution.

    Parameters
    ----------
    y : ndarray
        Target values.

    lower : float, optional
        Lower threshold.

    upper : float, optional
        Upper threshold.

    Returns
    -------
    ndarray of bool
        Boolean mask selecting rare observations.
    """

    mask = np.zeros(len(y), dtype=bool)

    if lower is not None:
        mask |= y <= lower

    if upper is not None:
        mask |= y >= upper

    return mask

def relevance_function():
    """
    Build the relevance function φ(y).

    Notes
    -----
    This function will be implemented following the utility-based
    regression framework proposed by Torgo and Ribeiro and later
    adopted by SMOGN.
    """
    raise NotImplementedError


def phi():
    """
    Compute the relevance score φ(y).

    Returns
    -------
    ndarray
        Relevance scores in [0, 1].
    """
    raise NotImplementedError


def automatic_phi():
    """
    Automatically estimate the relevance function from the target
    distribution.

    This implementation will follow the automatic relevance
    estimation procedure used in SMOGN.
    """
    raise NotImplementedError