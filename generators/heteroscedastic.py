import numpy as np
from typing import Optional, Tuple
from .base import SyntheticGenerator
from .function import *

class HeteroscedasticGenerator(SyntheticGenerator):
    """
    Generate synthetic regression datasets with heteroscedastic noise.

    The predictors are sampled independently from U(0, 1),
    and the target is generated as

        y = f(X) + ε(X)

    where f is a configurable function and
    ε(X) is Gaussian noise with variance that depends on X.

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
        