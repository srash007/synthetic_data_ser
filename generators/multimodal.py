import numpy as np
from typing import Optional, Tuple
from .base import SyntheticGenerator
from .function import *

class MultiModalGenerator(SyntheticGenerator):
    """
    Generate synthetic regression datasets with multimodal target distributions.

    The predictors are sampled independently from U(0, 1),
    and the target is generated as

        y = f(X) + ε

    where f is a configurable function and
    ε is Gaussian noise.

    Supported functions:
        - "linear"
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
        proportion: float = (0.5, 0.5),
        n_samples: int = 1000,
        n_features: int = 1,
        noise: float = 0.1,
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state)
        self.function = function.lower()
        self.proportion = proportion
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.n_modes = len(proportion)  # Number of modes is determined by the length of the proportion tuple
        
        
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        
        """
        Generate a synthetic multimodal regression dataset.

        Returns:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
        """
        
        # Generate predictor variables.
        X = self.rng.random((self.n_samples, self.n_features))
        x = X[:, 0]

        # Initialize the target vector.
        y = np.zeros(self.n_samples)

        # Randomly assign each sample to one of the modes.
        labels = self.rng.choice(
            self.n_modes,
            size=self.n_samples,
            p=self.proportion,
            
        )
        functions = FUNCTIONS[self.function]
        # Generate target values independently for each mode.
        for i, f in enumerate(self.functions):
            mask = labels == i
            y[mask] = f(x[mask])

        # Add Gaussian noise to the target values.
        y += self.rng.normal(
            0,
            self.noise,
            self.n_samples,
        )

        return X, y
        