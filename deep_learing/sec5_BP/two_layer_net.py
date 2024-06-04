import os
import sys
from collections import OrderedDict
from typing import Dict

import numpy as np
from common import numerical_gradient

# from common import numerical_gradient
from layer import Affine, Relu, SoftmaxWithLoss

sys.path.append(os.pardir)


class TwoLayerNet:

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weight_init_std: float = 0.01,
    ) -> None:
        # Initialize weights and biases
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(
            input_size, hidden_size
        )
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(
            hidden_size, output_size
        )
        self.params["b2"] = np.zeros(output_size)

        # Create layers
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        return float(self.last_layer.forward(y, t))

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(
        self, x: np.ndarray, t: np.ndarray
    ) -> Dict[str, np.ndarray]:

        def loss_w(w: np.ndarray) -> float:
            return self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_w, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_w, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_w, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_w, self.params["b2"])

        return grads

    def gradient(self, x: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads["W1"], grads["b1"] = (
            self.layers["Affine1"].dW,
            self.layers["Affine1"].db,
        )
        grads["W2"], grads["b2"] = (
            self.layers["Affine2"].dW,
            self.layers["Affine2"].db,
        )

        return grads
