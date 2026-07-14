import numpy as np
import random 


def generate_linear(
    n_sample: int ,
    noise: float ,
    random_seed: int ,
):
    np.random.seed(random_seed)
    X = np.random.uniform(-5, 5, n_sample)
    y = 2 * X + np.random.normal(0, noise, len(X))
    return X.reshape(-1, 1), y

def generate_quadratic(
    n_sample: int,
    noise: float,
    random_seed: int ,
):
    np.random.seed(random_seed)
    X = np.random.uniform(-5, 5, n_sample)
    y = X ** 2 + np.random.normal(0, noise, len(X))
    return X.reshape(-1, 1), y

def generate_exponential(
    n_sample: int ,
    noise: float,
    random_seed: int ,
):
    np.random.seed(random_seed)
    X = np.random.uniform(-5, 5, n_sample)
    y = np.exp(X) + np.random.normal(0, noise, len(X))
    return X.reshape(-1, 1), y

def generate_sinusoidal(
    n_sample: int ,
    noise: float ,
    random_seed: int ,
):
    np.random.seed(random_seed)
    X = np.random.uniform(-5, 5, n_sample)
    y = np.sin(X) + np.random.normal(0, noise, len(X))
    return X.reshape(-1, 1), y

def generate_piecewise(
    n_sample: int ,
    noise: float ,
    random_seed: int ,
):
    np.random.seed(random_seed)
    X = np.random.uniform(-5, 5, n_sample)
    y = np.where(X < 0, 2 * X + np.random.normal(0, noise, len(X)), X ** 2 + np.random.normal(0, noise, len(X)))
    return X.reshape(-1, 1), y

def generate_data(
    relation="linear",
    n_samples=1000,
    noise=1.0,
    random_seed=42,
):
    if relation == "linear":
        return generate_linear(n_samples, noise, random_seed)

    elif relation == "quadratic":
        return generate_quadratic(n_samples, noise, random_seed)

    elif relation == "exponential":
        return generate_exponential(n_samples, noise, random_seed)

    elif relation == "sinusoidal":
        return generate_sinusoidal(n_samples, noise, random_seed)

    elif relation == "piecewise":
        return generate_piecewise(n_samples, noise, random_seed)

    else:
        raise ValueError("Unknown relation")
    


def make_imbalanced(X, y, imbalance_ratio=0.1, random_seed=42):
    np.random.seed(random_seed)
    n_samples = len(y)
    n_minority = int(n_samples * imbalance_ratio)
    n_majority = n_samples - n_minority

    # Get indices of majority and minority classes
    majority_indices = np.where(y >= 0)[0]
    minority_indices = np.where(y < 0)[0]

    # Randomly select samples from majority class
    selected_majority_indices = np.random.choice(majority_indices, n_majority, replace=False)

    # Combine selected majority samples with all minority samples
    new_indices = np.concatenate([selected_majority_indices, minority_indices])
    np.random.shuffle(new_indices)

    return X[new_indices], y[new_indices]