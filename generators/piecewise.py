import numpy as np
from typing import Optional, Tuple
from .base import SyntheticGenerator
from typing import Callable
from .function import *

class PiecewiseGenerator(SyntheticGenerator):
    """
    Generate synthetic piecewise regression datasets.

    The predictors are sampled independently from U(0, 1),
    and the target is generated as a piecewise function of the predictors
    with added Gaussian noise.
    """

    def __init__(
        self,
        functions= [linear, cubic, sine],
        n_samples: int = 1000,
        n_features: int = 1,
        noise: float = 0.1,
        breakpoints: Tuple[float, float] = (0.3, 0.7),
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state)
        self.functions = self.functions = functions
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.breakpoints = breakpoints

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic piecewise regression dataset.

        Returns:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
        """
        
        X = self.rng.random((self.n_samples, self.n_features))
        x = X[:, 0]
        
        y = np.zeros(self.n_samples)

        b1, b2 = self.breakpoints

        mask1 = x < b1
        mask2 = (x >= b1) & (x < b2)
        mask3 = x >= b2

        y[mask1] = self.functions[0](x[mask1])
        y[mask2] = self.functions[1](x[mask2])
        y[mask3] = self.functions[2](x[mask3])

        y += self.rng.normal(
            0,
            self.noise,
            self.n_samples
        )

        return X, y