import numpy as np
from typing import Optional, Tuple
from .base import SyntheticGenerator
from .function import *


class FunctionGenerator(SyntheticGenerator):
    """
    Generate synthetic regression datasets based on mathematical functions.

    The predictors are sampled independently from U(0, 1),
    and the target is generated as

        y = f(X) + ε

    where f is a configurable function and
    ε is Gaussian noise.

    Supported functions:
        - "linear"
        - "quadratic"
        - "cubic"
        - "polynomial"
        - "sine"
        - "cosine"
        - "exponential"
        - "logarithm"
        - "sigmoid"
        - "absolute"
        - "step"
    """

    AVAILABLE_FUNCTIONS = (
        "linear",
        "quadratic",
        "cubic",
        "polynomial",
        "sine",
        "cosine",
        "exponential",
        "logarithm",
        "sigmoid",
        "absolute",
        "step",
    )

    def __init__(
        self,
        function: str = "linear",
        n_samples: int = 1000,
        n_features: int = 1,
        noise: float = 0.1,
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state)
        self.function = function.lower()
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        
        if self.function not in self.AVAILABLE_FUNCTIONS:
            raise ValueError(
                f"Unknown function '{self.function}'. "
                f"Choose one of {self.AVAILABLE_FUNCTIONS}."
            )
            
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic regression dataset based on the specified function.

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target vector of shape (n_samples,).
        """
        X = self.rng.random((self.n_samples, self.n_features))
        noise = self.rng.normal(0, self.noise, self.n_samples)
        
        if self.function == "linear":
            y = linear(X) + noise
        elif self.function == "cubic":
            y = cubic(X) + noise
        elif self.function == "polynomial":
            y = polynomial(X) + noise
        elif self.function == "sine":
            y = sine(X) + noise
        elif self.function == "cosine":
            y = cosine(X) + noise
        elif self.function == "exponential":
            y = exponential(X) + noise
        elif self.function == "logarithm":
            y = logarithm(X) + noise
        elif self.function == "sigmoid":
            y = sigmoid(X) + noise
        elif self.function == "absolute":
            y = absolute(X) + noise
        elif self.function == "step":
            y = step(X) + noise
        
        return X, y