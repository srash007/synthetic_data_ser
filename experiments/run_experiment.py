"""
Generic experiment runner.

Author
------
Sarah Elyane Rashiwa

Description
-----------
Run a complete synthetic regression experiment.

Pipeline
--------
Generator
    ↓
Train/Test Split
    ↓
Optional Preprocessing
    ↓
Regression Model
    ↓
Prediction
    ↓
Evaluation
"""

from __future__ import annotations

import pandas as pd


from metrics import regression_metrics


def run_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    model,
    preprocessing=None,
):
    """
    Run a complete regression experiment.

    Parameters
    ----------


    model
        Regression model implementing fit() and predict().

    preprocessing
        Optional preprocessing implementing fit_resample().

    Returns
    -------
    dict
        Experiment results.
    """

    # ---------------------------------------
    # Optional preprocessing
    # ---------------------------------------

    if preprocessing is not None:

        X_train, y_train = preprocessing.fit_resample(
            X_train,
            y_train,
        )

    # ---------------------------------------
    # Train model
    # ---------------------------------------

    model.fit(X_train, y_train)

    # ---------------------------------------
    # Prediction
    # ---------------------------------------

    y_pred = model.predict(X_test)

    # ---------------------------------------
    # Metrics
    # ---------------------------------------

    metrics = regression_metrics(
        y_test,
        y_pred,
    )

    # ---------------------------------------
    # Return
    # ---------------------------------------

    return {


        "preprocessing":
            None if preprocessing is None
            else preprocessing.__class__.__name__,

        "model": model.__class__.__name__,

        **metrics,
        
    }