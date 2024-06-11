import argparse
import glob
import os
import sys

sys.path.insert(
    0, f"{os.environ['HOME']}/dev-root/opencv4/lib/python3.10/site-packages"
)
import cv2
import matplotlib.pyplot as plt  # グラフ出力用module
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from model import Net
from torchvision.transforms import functional as TF

PATH = "./deep_learning_ws/deep_learning/torch_mnist/"
pred_data_path = os.path.join(PATH, "pred_data")

# road model
net = Net(0.001)
net.load_state_dict(torch.load(os.path.join(PATH, "model", "model.pth")))
net.eval()

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)

# road picture data for pred data
# 画像ファイル名を全て取得
img_paths = os.path.join(pred_data_path, "*.png")
img_path_list = glob.glob(img_paths)

# 画像データ・正解ラベル格納用配列
data = []

# 各画像データ・正解ラベルを格納する
for img_path in img_path_list:
    # 画像読み込み、28*28ピクセルでグレースケール変換
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = TF.to_tensor(img)

    # 画像をdataにセット
    data.append(img.detach().numpy())  # 配列にappendするため、一度ndarray型へ

# PyTorchで扱うため、tensor型にする
data = np.array(data)
data = torch.tensor(data)

# # 画像データ・正解ラベルのペアをデータにセットする
# dataset = torch.utils.data.TensorDataset(data)

# セットしたデータをバッチサイズごとの配列に入れる。
loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False)

# データ数を取得
data_size = len(img_path_list)

device = torch.device("cuda:0")
# net = net.to(device)

# prediction
inputs = data
# inputs = inputs.to(device)
outputs = net(inputs)
_, predicted = torch.max(outputs, 1)

# 各画像データに対する予測結果を出力
for i, img_path in enumerate(img_path_list):
    print("predict: ", predicted[i].item(), "  ", img_path)
    plt.imshow(cv2.imread(img_path))
    plt.show()
