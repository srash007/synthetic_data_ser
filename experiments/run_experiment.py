import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .synthetic_data import *

N_SAMPLE = 1000
NOISE = 2
RANDOM_SEED = 42
TEST_SIZE = 0.3
X, y = generate_linear(n_sample=N_SAMPLE, noise=NOISE, random_seed=RANDOM_SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)


# prepare dataframe for smoter: smogn expects a single DataFrame and the name of the target column
df_train = pd.concat([
	pd.DataFrame(X_train).reset_index(drop=True),
	pd.Series(y_train, name="target").reset_index(drop=True)
], axis=1)

# apply smoting for regression
df_smogn = smoter(data=df_train, y='target', random_state=RANDOM_SEED)
