"""
Collection of mathematical functions used by synthetic
regression data generators.
"""

import numpy as np


def linear(x, slope=1.0, intercept=0.0):
    """Linear function."""
    return slope * x + intercept

def cubic(x):
    """Cubic function."""
    return x**3


def polynomial(x, coefficients=(1.0, -2.0, 3.0)):
    """General polynomial."""
    return (
        coefficients[0] * x**2
        + coefficients[1] * x
        + coefficients[2]
    )


def sine(x, amplitude=1.0, frequency=2*np.pi):
    """Sine function."""
    return amplitude * np.sin(frequency * x)


def cosine(x, amplitude=1.0, frequency=2*np.pi):
    """Cosine function."""
    return amplitude * np.cos(frequency * x)


def exponential(x, scale=2.0):
    """Exponential function."""
    return np.exp(scale * x)


def logarithm(x, scale=5.0):
    """Logarithmic function."""
    return np.log1p(scale * x)

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

def absolute(x):
    """Absolute value function."""
    return np.abs(x)

def step(x, threshold=0.5):
    """Step function."""
    return np.where(x < threshold, 0, 1)

FUNCTIONS = {
    "linear": linear,
    "polynomial": polynomial,
    "cubic": cubic,
    "sine": sine,
    "cosine": cosine,
    "exponential": exponential,
    "logarithm": logarithm,
    "sigmoid": sigmoid,
    "absolute": absolute,
    "step": step
}

