import numpy as np
import pandas as pd

from .regression import *



def regression_metrics(y_true, y_pred):
    """
    Compute standard regression metrics.

    Returns
    -------
    dict
    """
    return {
        "R2": r2(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
    }