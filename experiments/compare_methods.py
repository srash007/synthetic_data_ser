"""
Compare methods.
"""

import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
)


from sklearn.model_selection import train_test_split


from generators import FunctionGenerator
from preprocessing.smogn import SMOGNPreprocessor
from ser import SERRegressor

from .run_experiment import run_experiment


generator = FunctionGenerator(
    function="linear",
    n_samples=1000,
    n_features=1,
    random_state=42
)

X,y = generator.generate()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


results = []

results.append(run_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    model=SERRegressor("B"),
))

results.append(run_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    model=LinearRegression(),
))

results.append(run_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    model=LinearRegression(),
    preprocessing=SMOGNPreprocessor(
        rel_thres=0.80,
    ),
))

df = pd.DataFrame(results)

print(df)