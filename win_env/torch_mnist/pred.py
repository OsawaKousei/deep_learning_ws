import glob
import os
import sys

import numpy as np
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, resize  # 画像処理用module
from model import Net
from torch import load, max, tensor, utils, device
from torchvision import transforms

PATH = "./"
pred_data_path = os.path.join(PATH, "pred_data")

# load model
net = Net(0.001)
net.load_state_dict(load(os.path.join(PATH, "model", "model.pth"), map_location=device('cpu')))
net.eval()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# load picture data for pred data
# get file path list
img_paths = os.path.join(pred_data_path, "*.png")
img_path_list = glob.glob(img_paths)

data = []

# read image data and set to data
for img_path in img_path_list:
    # resize img to 28x28 pixels and convert to gray scale and tensor
    img = imread(img_path)
    img = cvtColor(img, COLOR_BGR2GRAY)
    img = resize(img, (28, 28))
    img = transforms.functional.to_tensor(img)

    data.append(img.detach().numpy())  # 配列にappendするため、一度ndarray型へ

# PyTorchで扱うため、tensor型にする
data = np.array(data)
data = tensor(data)

# # 画像データ・正解ラベルのペアをデータにセットする
# dataset = torch.utils.data.TensorDataset(data)

# セットしたデータをバッチサイズごとの配列に入れる。
loader = utils.data.DataLoader(data, batch_size=128, shuffle=False)

# データ数を取得
data_size = len(img_path_list)

# device = torch.device("cuda:0")
# net = net.to(device)

# prediction
inputs = data
# inputs = inputs.to(device)
outputs = net(inputs)
_, predicted = max(outputs, 1)

# 各画像データに対する予測結果を出力
for i, img_path in enumerate(img_path_list):
    print(predicted[i].item())
    # plt.imshow(cv2.imread(img_path))
    # plt.show()
