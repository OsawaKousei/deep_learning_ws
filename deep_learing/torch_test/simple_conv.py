import matplotlib.pyplot as plt  # import the library to draw the graph
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

BATCH_SIZE = 100
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.001
EPOCH = 20
PATH = "dataset"

# transform the data
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),  # convert the data to a tensor
        torchvision.transforms.Normalize((0.5,), (0.5,)),  # normalize the data
    ]
)

trainset = torchvision.datasets.MNIST(
    root=PATH, train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

testset = torchvision.datasets.MNIST(
    root=PATH, train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(4320, 100)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


device = torch.device("cuda:0")
net = Net()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

train_loss_value = []  # list to hold the training loss
train_acc_value = []  # list to hold the training accuracy
test_loss_value = []  # list to hold the test loss
test_acc_value = []  # list to hold the test accuracy

for epoch in range(EPOCH):
    print("epoch", epoch + 1)  # print epoch number
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0

    # check the accuracy and loss of the network using the training data
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()  # add the loss
        _, predicted = outputs.max(1)  # get the position of the maximum value
        sum_total += labels.size(
            0
        )  # add the number of data in the batch to the total number of data
        sum_correct += (
            (predicted == labels).sum().item()
        )  # add the number of correct answers \
        # to the total number of correct answers
    print(
        "train mean loss={}, accuracy={}".format(
            sum_loss * BATCH_SIZE / len(trainloader.dataset),
            float(sum_correct / sum_total),
        )
    )  # print the average loss and accuracy of the training data
    train_loss_value.append(
        sum_loss * BATCH_SIZE / len(trainloader.dataset)
    )  # record the average loss of the training data
    train_acc_value.append(
        float(sum_correct / sum_total)
    )  # record the accuracy of the training data

    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0

    # check the accuracy and loss of the network using the test data
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        _, predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted == labels).sum().item()
    print(
        "test  mean loss={}, accuracy={}".format(
            sum_loss * BATCH_SIZE / len(testloader.dataset),
            float(sum_correct / sum_total),
        )
    )
    test_loss_value.append(sum_loss * BATCH_SIZE / len(testloader.dataset))
    test_acc_value.append(float(sum_correct / sum_total))

plt.figure(figsize=(6, 6))  # set the size of the graph

# draw the graph
plt.plot(range(EPOCH), train_loss_value)
plt.plot(range(EPOCH), test_loss_value, c="#00ff00")
plt.xlim(0, EPOCH)
plt.ylim(0, 2.5)
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.legend(["train loss", "test loss"])
plt.title("loss")
plt.savefig("loss_image.png")
plt.clf()

plt.plot(range(EPOCH), train_acc_value)
plt.plot(range(EPOCH), test_acc_value, c="#00ff00")
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY")
plt.legend(["train acc", "test acc"])
plt.title("accuracy")
plt.savefig("accuracy_image.png")
plt.show()
