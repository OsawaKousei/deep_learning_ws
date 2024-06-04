from typing import Optional, Tuple

import numpy as np
from common import cross_entropy_error, softmax


class MullLayer:
    def __init__(self) -> None:
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        out = x * y

        return out  # type: ignore

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = x + y
        return out  # type: ignore

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu:
    def __init__(self) -> None:
        self.mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out  # type: ignore

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout * (1.0 - self.out) * self.out  # type: ignore

        return dx  # type: ignore


class Affine:
    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        self.W = W
        self.b = b
        self.x: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out  # type: ignore

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)  # type: ignore
        self.db = np.sum(dout, axis=0)

        return dx  # type: ignore


class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.loss: Optional[float] = None
        self.y: Optional[np.ndarray] = None
        self.t: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss  # type: ignore

    def backward(self, dout: float = 1) -> np.ndarray:
        batch_size = self.t.shape[0]  # type: ignore
        dx = (self.y - self.t) / batch_size  # type: ignore

        return dx  # type: ignore
