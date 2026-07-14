import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from ser import SERRegressor


# =====================================================
# Generate synthetic data
# =====================================================

np.random.seed(42)

X = pd.DataFrame({
    "x": np.random.uniform(-5, 5, 5000)
})

y = (
    X["x"] ** 2
    + np.random.normal(0, 2, len(X))
)

# =====================================================
# Train / Test split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
)

# =====================================================
# Train SER
# =====================================================

ser = SERRegressor(
    segmentation="A",
    verbose=True,
)

result = ser.fit_predict(
    X_train,
    y_train,
    X_test,
    y_test,
)

print("\n========== Metrics ==========")
print(result["metrics"])

print("\n========== Selected Experts ==========")

for group, expert in result["local_experts"].items():

    print(group, "->", expert["name"])