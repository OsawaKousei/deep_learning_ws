import os
import sys

import common
import numpy as np

sys.path.append(os.pardir)


# two layer neural network
class TwoLayerNet:
    # constructor
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weight_init_std: float = 0.01,
    ) -> None:
        # initialize weights
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(
            input_size, hidden_size
        )
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(
            hidden_size, output_size
        )
        self.params["b2"] = np.zeros(output_size)

    # predict
    def predict(self, x: np.ndarray) -> np.ndarray:
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = common.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y: np.ndarray = common.softmax(a2)

        return y

    # loss
    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)

        return float(common.cross_entropy_error(y, t))

    # accuracy
    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # loss function for weight
    # def loss_W(self, x: np.ndarray, t: np.ndarray) -> float:
    #     return self.loss(x, t)

    # numerical gradient
    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        print(f"x.shape: {x.shape}")
        print(f"self.params['W1'].shape: {self.params['W1'].shape}")
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads["W1"] = common.numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = common.numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = common.numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = common.numerical_gradient(loss_W, self.params["b2"])

        return grads
