from generators import FunctionGenerator
from generators.heteroscedastic import HeteroscedasticGenerator
from generators.multimodal import MultiModalGenerator
from generators.piecewise import PiecewiseGenerator

from ser import SERRegressor
from metrics.metrics import rmse, mae, r2_score

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ============================================================
# Generators to test
# ============================================================

generators = [
    FunctionGenerator("linear"),
    FunctionGenerator("cubic"),
    FunctionGenerator("polynomial"),
    FunctionGenerator("sine"),
    FunctionGenerator("cosine"),
    FunctionGenerator("exponential"),
    FunctionGenerator("logarithm"),
    FunctionGenerator("sigmoid"),
    FunctionGenerator("absolute"),
    FunctionGenerator("step"),

    PiecewiseGenerator(),
    HeteroscedasticGenerator(),  
    MultiModalGenerator(),
]


# ============================================================
# Utility functions
# ============================================================

def generator_name(generator):
    """Return a readable generator name."""

    if isinstance(generator, FunctionGenerator):
        return f"FunctionGenerator ({generator.function})"

    return generator.__class__.__name__


def print_metrics(name, y_true, y_pred):
    print(
        f"{name:5} | "
        f"RMSE = {rmse(y_true, y_pred):.4f} | "
        f"MAE = {mae(y_true, y_pred):.4f} | "
        f"R² = {r2_score(y_true, y_pred):.4f}"
    )


# ============================================================
# Smoke test for one generator
# ============================================================

def run_generator(generator):

    print("=" * 70)
    print(f"Testing {generator_name(generator)}")
    print("=" * 70)

    # --------------------------------------------------------
    # Generate data
    # --------------------------------------------------------

    X, y = generator.generate()

    assert X.shape == (1000, 1)
    assert y.shape == (1000,)

    assert np.isfinite(X).all()
    assert np.isfinite(y).all()

    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
    )

    # --------------------------------------------------------
    # Train OLS
    # --------------------------------------------------------

    ols = LinearRegression()

    ols.fit(X_train, y_train)

    # --------------------------------------------------------
    # Train SER
    # --------------------------------------------------------

    ser = SERRegressor("A")

    ser.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred_ols = ols.predict(X_test)
    y_pred_ser = ser.predict(X_test)

    assert y_pred_ols.shape == y_test.shape
    assert y_pred_ser.shape == y_test.shape

    assert np.isfinite(y_pred_ols).all()
    assert np.isfinite(y_pred_ser).all()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print_metrics("OLS", y_test, y_pred_ols)
    print_metrics("SER", y_test, y_pred_ser)

    print("PASS\n")


# ============================================================
# Main
# ============================================================

def main():

    passed = 0
    failed = 0

    for generator in generators:

        try:

            run_generator(generator)
            passed += 1

        except Exception as e:

            failed += 1

            print(f"FAIL ({generator_name(generator)})")
            print(e)
            print()

    print("=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)

    print(f"Passed : {passed}")
    print(f"Failed : {failed}")

    if failed == 0:
        print("\nALL GENERATORS PASSED")
    else:
        print("\nSOME GENERATORS FAILED")


if __name__ == "__main__":
    main()