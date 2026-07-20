import numpy as np
from typing import Optional, Tuple
from .base import SyntheticGenerator
from .function import *

class ImbalanceGenerator(SyntheticGenerator):
    """
    Generate synthetic regression datasets with imbalanced target distributions.

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
        imbalance_ratio: float = 0.5,
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state)
        self.function = function.lower()
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.imbalance_ratio = imbalance_ratio
    