import numpy as np
import random 

# seed = random.randint(0, 10000)
RANDOM_SEED = 42
N_SAMPLE = 1000
NOISE = 2

def generate_linear(
    n_sample: int = N_SAMPLE,
    noise: float = NOISE,
    random_seed: int = RANDOM_SEED,
):
    np.random.seed(random_seed)
    X = np.random.uniform(-5, 5, n_sample)
    y = 2 * X + np.random.normal(0, noise, len(X))
    return X.reshape(-1, 1), y

def generate_quadratic(
    n_sample: int = N_SAMPLE,
    noise: float = NOISE,
    random_seed: int = RANDOM_SEED,
):
    np.random.seed(random_seed)
    X = np.random.uniform(-5, 5, n_sample)
    y = X ** 2 + np.random.normal(0, noise, len(X))
    return X.reshape(-1, 1), y

def generate_exponential(
    n_sample: int = N_SAMPLE,
    noise: float = NOISE,
    random_seed: int = RANDOM_SEED,
):
    np.random.seed(random_seed)
    X = np.random.uniform(-5, 5, n_sample)
    y = np.exp(X) + np.random.normal(0, noise, len(X))
    return X.reshape(-1, 1), y

def generate_sinusoidal(
    n_sample: int = N_SAMPLE,
    noise: float = NOISE,
    random_seed: int = RANDOM_SEED,
):
    np.random.seed(random_seed)
    X = np.random.uniform(-5, 5, n_sample)
    y = np.sin(X) + np.random.normal(0, noise, len(X))
    return X.reshape(-1, 1), y

def generate_piecewise(
    n_sample: int = N_SAMPLE,
    noise: float = NOISE,
    random_seed: int = RANDOM_SEED,
):
    np.random.seed(random_seed)
    X = np.random.uniform(-5, 5, n_sample)
    y = np.where(X < 0, 2 * X + np.random.normal(0, noise, len(X)), X ** 2 + np.random.normal(0, noise, len(X)))
    return X.reshape(-1, 1), y
