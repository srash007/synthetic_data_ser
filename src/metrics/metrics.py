import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from ser.utils import to_numpy

def phi_rmse():
    return None

def metrics_table(y_true, y_pred, label):
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    return pd.DataFrame([{
        "Model": label,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }])