import numpy as np


def softmax(a: np.ndarray) -> np.ndarray:
    c: float = np.max(a)
    exp_a: np.ndarray = np.exp(a - c)  # prevent overflow
    sum_exp_a: float = np.sum(exp_a)
    y: np.ndarray = exp_a / sum_exp_a
    return y
