import torch

# simple model with 2 linear layers
model = torch.nn.Sequential(
    torch.nn.Linear(10, 100), torch.nn.ReLU(), torch.nn.Linear(100, 10)
)
model.cuda()

# Check if the model is on GPUW
print(next(model.parameters()).is_cuda)

# input tensor
input = torch.randn(1, 10).cuda()
output = model(input)

print(output)

# free up memory
del model
