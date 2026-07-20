from metrics.metrics import *
from generators import *
from ser import SERRegressor
from metrics.metrics import rmse, mae, r2_score

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


""" Generate dataset """
generator = FunctionGenerator("linear", 1000,1)
X,y = generator.generate()


X_train,X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.3, random_state=42
)
assert len(X_train) == 700
assert len(X_test) == 300

modelOLS = LinearRegression()
modelOLS.fit(X_train,y_train)

modelSER = SERRegressor("A")
modelSER.fit(X_train,y_train)

y_pred_ols = modelOLS.predict(X_test)
y_pred_ser = modelSER.predict(X_test)

assert y_pred_ols.shape == y_test.shape
assert y_pred_ser.shape == y_test.shape

print("\nOLS")

print("RMSE:", rmse(y_test, y_pred_ols))
print("MAE :", mae(y_test, y_pred_ols))
print("R²  :", r2_score(y_test, y_pred_ols))

print("\nSER")

print("RMSE:", rmse(y_test, y_pred_ser))
print("MAE :", mae(y_test, y_pred_ser))
print("R²  :", r2_score(y_test, y_pred_ser))

assert np.isfinite(y_pred_ols).all()
assert np.isfinite(y_pred_ser).all()

print("\n==============================")
print("Smoke test PASSED")
print("==============================")