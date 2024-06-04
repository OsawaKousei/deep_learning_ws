from typing import Callable, Tuple

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
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# numerical gradient
def numerical_gradient(f: Callable, x: np.ndarray) -> np.ndarray:
    h: float = 1e-4  # 0.0001
    grad: np.ndarray = np.zeros_like(x)

    it: np.nditer = np.nditer(
        x, flags=["multi_index"], op_flags=["readwrite"]  # type: ignore
    )

    while not it.finished:
        idx: Tuple[int, ...] = it.multi_index
        tmp_val: float = float(x[idx])
        x[idx] = tmp_val + h
        fxh1: np.ndarray = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2: np.ndarray = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # restore the original value
        it.iternext()

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


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
