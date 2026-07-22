"""
Compare baseline regression models.
"""

import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
)

from sklearn.ensemble import RandomForestRegressor

from generators import FunctionGenerator
from ser import SERRegressor

from .run_experiment import run_experiment

generator = FunctionGenerator(
    function="linear",
    n_samples=1000,
    n_features=1,
    random_state=42
)

models = [

    LinearRegression(),

    Ridge(),

    Lasso(),

    RandomForestRegressor(
        random_state=42
    ),

    SERRegressor("B",verbose=False),
]

results = []

for model in models:

    results.append(

        run_experiment(
            generator=generator,
            model=model,
        )

    )
df = pd.DataFrame(results)

print(df)

df.to_csv(
    "results/baselines.csv",
    index=False,
)