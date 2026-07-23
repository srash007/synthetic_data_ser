"""
Compare methods.
"""

import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
)


from sklearn.model_selection import train_test_split


from generators import FunctionGenerator
from preprocessing.smogn import SMOGNPreprocessor
from ser import SERRegressor

from .run_experiment import run_experiment


generator = FunctionGenerator(
    function="cubic",
    n_samples=10000,
    n_features=20,
    random_state=42
)

X,y = generator.generate()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
thresholds = [0.85, 0.90, 0.95,0.99]

results = []

for threshold in thresholds:

    results.append(
        run_experiment(
            X_train,
            X_test,
            y_train,
            y_test,
            model=LinearRegression(),
            preprocessing=SMOGNPreprocessor(
                rel_thres=threshold
            ),
        )
    )

df = pd.DataFrame(results)

print(df)