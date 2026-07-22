"""
Metrics dedicated to imbalanced regression problems.

This module provides metrics specifically designed to evaluate
predictive performance on rare target values.

Unlike standard regression metrics (RMSE, MAE, R²), these metrics
focus only on observations belonging to the lower and/or upper tails
of the target distribution.

Author
------
Sarah Rashiwa
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from .utils import tail_mask


def tail_rmse(y_true, y_pred, lower=None, upper=None):
    """
    Compute the Root Mean Squared Error (RMSE) on rare observations.

    Rare observations are defined using the lower and/or upper
    thresholds.

    Parameters
    ----------
    y_true : ndarray
        True target values.

    y_pred : ndarray
        Predicted target values.

    lower : float, optional
        Lower threshold defining the lower tail.

    upper : float, optional
        Upper threshold defining the upper tail.

    Returns
    -------
    float
        Tail RMSE.
    """

    mask = tail_mask(y_true, lower, upper)

    if mask.sum() == 0:
        return np.nan

    return np.sqrt(
        mean_squared_error(
            y_true[mask],
            y_pred[mask],
        )
    )


def tail_mae(y_true, y_pred, lower=None, upper=None):
    """
    Compute the Mean Absolute Error (MAE) on rare observations.
    """

    mask = tail_mask(y_true, lower, upper)

    if mask.sum() == 0:
        return np.nan

    return mean_absolute_error(
        y_true[mask],
        y_pred[mask],
    )


def tail_r2(y_true, y_pred, lower=None, upper=None):
    """
    Compute the coefficient of determination (R²)
    on rare observations only.
    """

    mask = tail_mask(y_true, lower, upper)

    if mask.sum() < 2:
        return np.nan

    return r2_score(
        y_true[mask],
        y_pred[mask],
    )


def tail_coverage(y_true, y_pred, lower=None, upper=None):
    """
    Compute the proportion of true rare observations that are
    correctly predicted as rare.

    Notes
    -----
    This metric is inspired by the concept of recall but is adapted
    to regression by comparing the predicted and true tail regions.
    Values range from 0 to 1.
    """

    true_mask = tail_mask(
        y_true,
        lower,
        upper,
    )

    pred_mask = tail_mask(
        y_pred,
        lower,
        upper,
    )

    if true_mask.sum() == 0:
        return np.nan

    return np.sum(
        true_mask & pred_mask
    ) / np.sum(true_mask)


def weighted_mae(y_true, y_pred, weights):
    """
    Compute the weighted Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : ndarray
        True target values.

    y_pred : ndarray
        Predicted target values.

    weights : ndarray
        Observation weights. These are typically obtained from a
        relevance function and assign higher importance to rare
        observations.

    Returns
    -------
    float
        Weighted MAE.
    """

    error = np.abs(
        y_true - y_pred
    )

    return np.sum(
        weights * error
    ) / np.sum(weights)


def weighted_rmse(y_true, y_pred, weights):
    """
    Compute the weighted Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : ndarray
        True target values.

    y_pred : ndarray
        Predicted target values.

    weights : ndarray
        Observation weights.

    Returns
    -------
    float
        Weighted RMSE.
    """

    mse = np.sum(
        weights * (y_true - y_pred) ** 2
    )

    mse /= np.sum(weights)

    return np.sqrt(mse)

def precision_phi(y_true, y_pred, relevance, threshold=0.8):
    """
    Compute the utility-based Precisionφ metric.

    Precisionφ measures the proportion of observations predicted
    as rare that are actually rare according to the relevance
    function φ(y).

    Parameters
    ----------
    y_true : ndarray
        True target values.

    y_pred : ndarray
        Predicted target values.

    relevance : callable or ndarray
        Relevance function φ(y) or precomputed relevance scores.

    threshold : float, default=0.8
        Relevance threshold above which an observation is
        considered rare.

    Returns
    -------
    float
        Precisionφ score.

    Notes
    -----
    This metric follows the utility-based evaluation framework
    proposed for imbalanced regression problems. It will be
    implemented using the common relevance function shared by
    SER and SMOGN.

    References
    ----------
    Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013).
    SMOTE for Regression.
    """
    raise NotImplementedError

def recall_phi(y_true, y_pred, relevance, threshold=0.8):
    """
    Compute the utility-based Recallφ metric.

    Recallφ measures the proportion of truly rare observations
    that are correctly identified as rare by the regression model.

    Parameters
    ----------
    y_true : ndarray
        True target values.

    y_pred : ndarray
        Predicted target values.

    relevance : callable or ndarray
        Relevance function φ(y) or precomputed relevance scores.

    threshold : float, default=0.8
        Relevance threshold above which an observation is
        considered rare.

    Returns
    -------
    float
        Recallφ score.

    Notes
    -----
    Recallφ evaluates the ability of a regression model to
    recover observations belonging to the rare regions of the
    target distribution.

    References
    ----------
    Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013).
    SMOTE for Regression.
    """
    raise NotImplementedError

def f1_phi(y_true, y_pred, relevance, threshold=0.8):
    """
    Compute the utility-based F1φ metric.

    F1φ is the harmonic mean of Precisionφ and Recallφ,
    providing a single measure of predictive performance
    on rare observations.

    Parameters
    ----------
    y_true : ndarray
        True target values.

    y_pred : ndarray
        Predicted target values.

    relevance : callable or ndarray
        Relevance function φ(y) or precomputed relevance scores.

    threshold : float, default=0.8
        Relevance threshold above which an observation is
        considered rare.

    Returns
    -------
    float
        F1φ score.

    Notes
    -----
    F1φ balances precision and recall in utility-based
    regression evaluation and is one of the standard
    metrics used in the imbalanced regression literature.

    References
    ----------
    Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013).
    SMOTE for Regression.
    """
    raise NotImplementedError