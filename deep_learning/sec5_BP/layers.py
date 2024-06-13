from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from common import cross_entropy_error, softmax


class MullLayer:
    def __init__(self) -> None:
        self.x: Optional[npt.NDArray[np.float64]] = None
        self.y: Optional[npt.NDArray[np.float64]] = None

    def forward(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        self.x = x
        self.y = y
        out = x * y

        return npt.NDArray[np.float64](out)

    def backward(
        self, dout: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self) -> None:
        pass

    def forward(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        out = x + y
        return npt.NDArray[np.float64](out)

    def backward(
        self, dout: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu:
    def __init__(self) -> None:
        self.mask: npt.NDArray[np.float64] = np.zeros(0)

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(
        self, dout: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.out: npt.NDArray[np.float64] = np.zeros(0)

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = 1.0 / (1.0 + np.exp(-x))
        self.out = out

        return out

    def backward(
        self, dout: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        self.out = npt.NDArray[np.float64](self.out)
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(
        self, W: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
    ) -> None:
        self.W: npt.NDArray[np.float64] = W
        self.b: npt.NDArray[np.float64] = b
        self.x: npt.NDArray[np.float64] = np.zeros(0)
        self.dW: npt.NDArray[np.float64] = np.zeros(0)
        self.db: npt.NDArray[np.float64] = np.zeros(0)

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(
        self, dout: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.loss: float = 0.0
        self.y: npt.NDArray[np.float64] = np.zeros(0)
        self.t: npt.NDArray[np.float64] = np.zeros(0)

    def forward(
        self, x: npt.NDArray[np.float64], t: npt.NDArray[np.float64]
    ) -> float:
        self.t = t
        self.y = softmax(x)
        self.loss = float(cross_entropy_error(self.y, self.t))

        return self.loss

    def backward(self, dout: float = 1) -> npt.NDArray[np.float64]:
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
