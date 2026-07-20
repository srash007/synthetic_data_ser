from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class SyntheticGenerator(ABC):
    """
    Abstract base class for synthetic regression generators.

    Every generator must implement the generate() method and
    return a tuple (X, y).
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    @abstractmethod
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic regression dataset."""
        pass