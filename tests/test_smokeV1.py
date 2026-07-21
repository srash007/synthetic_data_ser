from generators import *
from ser import SERRegressor
from metrics.metrics import rmse, mae, r2_score

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def print_metrics(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"RMSE : {rmse(y_true, y_pred):.4f}")
    print(f"MAE  : {mae(y_true, y_pred):.4f}")
    print(f"R²   : {r2_score(y_true, y_pred):.4f}")


def main():

    # =====================================================
    # Generate synthetic dataset
    # =====================================================

    generator = FunctionGenerator("linear", 1000,1)
    X,y = generator.generate()

    assert X.shape == (1000, 1)
    assert y.shape == (1000,)

    assert np.isfinite(X).all()
    assert np.isfinite(y).all()

    # =====================================================
    # Train / Test split
    # =====================================================

    X_train,X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.3, random_state=42
    )
    assert len(X_train) == 700
    assert len(X_test) == 300

    # =====================================================
    # Train models
    # =====================================================

    modelOLS = LinearRegression()
    modelOLS.fit(X_train,y_train)

    modelSER = SERRegressor("B")
    modelSER.fit(X_train,y_train)

    assert hasattr(modelOLS, "coef_")
    assert modelSER is not None

    # =====================================================
    # Predictions
    # =====================================================

    y_pred_ols = modelOLS.predict(X_test)
    y_pred_ser = modelSER.predict(X_test)

    assert y_pred_ols.shape == y_test.shape
    assert y_pred_ser.shape == y_test.shape

    # =====================================================
    # Evaluation
    # =====================================================

    print_metrics("OLS", y_test, y_pred_ols)
    print_metrics("SER", y_test, y_pred_ser)


    print("\n==============================")
    print("Smoke test PASSED")
    print("==============================")
    
if __name__ == "__main__":
    main()