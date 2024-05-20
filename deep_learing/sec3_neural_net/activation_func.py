from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def step_function(x: np.ndarray) -> Any:
    return np.array(x > 0, dtype=np.int32)


def sigmoid(x: np.ndarray) -> Any:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> Any:
    return np.maximum(0, x)


# x = np.arange(-5.0, 5.0, 0.1)
# y = relu(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()
