import os
import pickle
import sys
from typing import Tuple

from activation_func import sigmoid
from softmax import softmax

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


# Display the first image in the training data
def img_show(img: np.ndarray) -> None:
    pil_img: Image.Image = Image.fromarray(np.uint8(img))
    pil_img.show()


# Load the MNIST dataset
(x_train, t_train), (x_test, t_test) = load_mnist(
    flatten=True, normalize=False
)


# Print the shape of the training data
def print_shapes() -> None:
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)


# Display the first image in the training data
def show_test() -> None:
    img = x_train[0]
    label = t_train[0]
    print(label)

    print
    img = img.reshape(28, 28)
    print(img.shape)

    img_show(img)


# get mnist data
def get_data() -> Tuple[np.ndarray, np.ndarray]:
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test


# initialize the network
def init_network() -> dict:
    with open("sample_weight.pkl", "rb") as f:
        network: dict = pickle.load(f)
    return network


# predict the number
def predict(network: dict, x: np.ndarray) -> np.ndarray:
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1: np.ndarray = np.dot(x, W1) + b1
    z1: np.ndarray = sigmoid(a1)
    a2: np.ndarray = np.dot(z1, W2) + b2
    z2: np.ndarray = sigmoid(a2)
    a3: np.ndarray = np.dot(z2, W3) + b3
    y: np.ndarray = softmax(a3)

    return y


# predict the number
x, t = get_data()
network = init_network()

accuracy_cnt: int = 0
for i in range(len(x)):
    y: np.ndarray = predict(network, x[i])
    p: int = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
