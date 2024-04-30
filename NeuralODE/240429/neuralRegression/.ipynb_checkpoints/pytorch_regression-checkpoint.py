import numpy as np
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch import nn, optim
import matplotlib.pyplot as plt

# ################## Part 1: load data and create batch ##################
# Generate data and shuffle
N_total = 600
N_train = 500
x = torch.unsqueeze(torch.linspace(0, 1, N_total), dim=1)
r = torch.randperm(N_total)
x = x[r, :]
y = 0.2 + 0.4 * torch.pow(x, 2) + 0.3 * x * torch.sin(15 * x) + 0.05 * torch.cos(50 * x)


# plt.scatter(x, y)
# plt.show()


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.y = y
        self.x = x

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        y1 = self.y[idx]
        x1 = self.x[idx]
        return (x1, y1)


trainset = CustomDataset(x[0:N_train, :], y[0:N_train, :])
testset = CustomDataset(x[N_train:N_total, :], y[N_train:N_total, :])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=50)
test_loader = torch.utils.data.DataLoader(testset, batch_size=50)

# ################## Part 2: Define Model and initialize ##################
# This part need to be changed to define a new model
model = nn.Sequential(nn.Linear(1, 1024, bias=True),
                      nn.ReLU(),
                      # nn.Linear(128, 64, bias=True),
                      # nn.ReLU(),
                      nn.Linear(1024, 1, bias=True)
                      )
print(model)


# ############## This part can be changed to different initialization
# Initialize as 0

# Initialize as uniform [-1, 1]
# for p in model.parameters():
#     p.data.uniform_(-1, 1)
def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-1, 1)
        m.bias.data.uniform_(-1, 1)

# Initialize as normal
# m.bias.data.normal_(0, 1)
# m.weight.data.normal_(0, 0.03)

model.apply(init_weights)

# Initialize as normal
#
# ########################################################################

# ################## Part 3: Define Loss and optimizer ##################

# ######## This can be changed to different loss function, e.g., L2loss
# ######## and different optimization parameter，e.g. regularization, learning rata.
criterion = torch.nn.L2Loss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# ########################################################################


# ################## Part 4: Optimization ##################
def train_NN():
    model.train()
    for images, labels in train_loader:
        images = images.view(images.shape[0], -1)
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss


def test_NN(loader):
    model.eval()
    loss = 0
    for images, labels in loader:
        images = images.view(images.shape[0], -1)
        out = model(images)
        loss += criterion(out, labels)
    loss = loss / len(loader)
    return loss


N_epoch = 500
train_loss = np.zeros((N_epoch, 1))
test_loss = np.zeros((N_epoch, 1))
for epoch in range(N_epoch):
    loss1 = train_NN()
    train_loss[epoch, 0] = test_NN(train_loader)
    test_loss[epoch, 0] = test_NN(test_loader)
    print(
        f'Epoch: {epoch:03d}, train loss: {train_loss[epoch, 0]:.7f}, test loss: {test_loss[epoch, 0]:.7f}')

x_test = torch.unsqueeze(torch.linspace(0, 1, 1999), dim=1)
y_test = model(x_test)

plt.plot(x[0:N_total], y[0:N_total], 'bo')
plt.plot(x_test, y_test.detach().numpy(), 'r')
plt.show()
