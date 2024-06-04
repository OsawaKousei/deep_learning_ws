import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

torch.backends.cudnn.enabled = False

# download the FashionMNIST dataset
trainset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True
)

testset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# define the model
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(1, 6, 5)
        self.pool: nn.MaxPool2d = nn.MaxPool2d(2)
        self.conv2: nn.Conv2d = nn.Conv2d(6, 16, 5)
        self.fc1: nn.Linear = nn.Linear(16 * 4 * 4, 120)
        self.fc2: nn.Linear = nn.Linear(120, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# create an instance of the model
model = CNN().cuda()

# define the loss function
criterion = nn.CrossEntropyLoss()
# define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

# train the model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()  # move to GPU

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(
                "[%d, %5d] loss: %.3f"
                % (epoch + 1, i + 1, running_loss / 2000)
            )
            running_loss = 0.0

print("Finished Training")

# Get the first batch of training data
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Move the data to the GPU
images, labels = images.cuda(), labels.cuda()

outputs = model(images)
_, predicted = torch.max(outputs.data, 1)

correct = (predicted == labels).sum().item()
total = labels.size(0)

print(
    "Accuracy of the network on the test images: %.3f %%"
    % (100 * correct / total)
)
