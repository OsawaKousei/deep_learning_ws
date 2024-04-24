import matplotlib.pyplot as plt
import numpy as np


def step_function(x: np.ndarray) -> np.ndarray:
    return np.array(x > 0, dtype=np.int32)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
