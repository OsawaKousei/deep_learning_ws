from typing import Callable

import numpy as np


def step_function(x: np.ndarray) -> np.ndarray:
    return np.array(x > 0, dtype=np.int32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


# sum of square error
def sum_squared_error(y: np.ndarray, t: np.ndarray) -> float:
    return 0.5 * np.sum((y - t) ** 2)


# cross entropy error
def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
    delta: float = 1e-7
    batch_size: int = y.shape[0]
    return -float(
        np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
    )


# numerical gradient
def numerical_gradient(f: Callable, x: np.ndarray) -> np.ndarray:
    h: float = 1e-4
    grad: np.ndarray = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val: float = float(x[idx])
        x[idx] = tmp_val + h
        fxh1: float = f(x)

        x[idx] = tmp_val - h
        fxh2: float = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


# gradient descent
def gradient_descent(
    f: Callable, init_x: np.ndarray, lr: float, step_num: int
) -> np.ndarray:
    x: np.ndarray = init_x

    for i in range(step_num):
        grad: np.ndarray = numerical_gradient(f, x)
        x -= lr * grad

    return x


def softmax(a: np.ndarray) -> np.ndarray:
    c: float = np.max(a)
    exp_a: np.ndarray = np.exp(a - c)  # prevent overflow
    sum_exp_a: float = np.sum(exp_a)
    y: np.ndarray = exp_a / sum_exp_a
    return y
