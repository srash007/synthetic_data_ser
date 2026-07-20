import numpy as np
from metrics.metrics import regression_metrics




y_true = np.array([1,2,3,4])

y_pred = np.array([1,2,3,4])

results = regression_metrics(y_true, y_pred)

assert results["RMSE"] == 0
assert results["MAE"] == 0
assert results["R2"] == 1
    

print(results)