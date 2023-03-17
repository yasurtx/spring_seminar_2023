from time import time

import numpy as np
import torch  
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import torchsummary

use_cuda = torch.cuda.is_available()
print('Use CUDA:', use_cuda)train_data = torchvision.datasets.MNIST(root="./", train=True, transform=transforms.ToTensor(), download=True)
test_data =  torchvision.datasets.MNIST(root="./", train=False, transform=transforms.ToTensor(), download=True)

print(type(train_data.data), type(train_data.targets))
print(type(test_data.data), type(test_data.targets))
print(train_data.data.size(), train_data.targets.size())
print(test_data.data.size(), test_data.targets.size())

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.l1 = nn.Linear(7*7*32, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 10)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        h = self.pool(self.act(self.conv1(x)))
        h = self.pool(self.act(self.conv2(h)))
        h = h.view(h.size()[0], -1)
        h = self.act(self.l1(h))
        h = self.act(self.l2(h))
        h = self.l3(h)
        return h
      
model = CNN()
if use_cuda:
    model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

if use_cuda:  
    torchsummary.summary(model, (1, 28, 28), device='cuda')
else:         
    torchsummary.summary(model, (1, 28, 28), device='cpu')
    
batch_size = 100
epoch_num = 10

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss() #～～のように工夫しました
if use_cuda:
    criterion.cuda()

model.train()

train_start = time()
for epoch in range(1, epoch_num+1):
    sum_loss = 0.0
    count = 0

    for image, label in train_loader:

        if use_cuda:
            image = image.cuda()
            label = label.cuda()

        y = model(image)

        loss = criterion(y, label)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        pred = torch.argmax(y, dim=1)
        count += torch.sum(pred == label)

    print("epoch: {}, mean loss: {}, mean accuracy: {}, elapsed time: {}".format(epoch, sum_loss/len(train_loader), count.item()/len(train_data), time() - train_start))
    
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

model.eval()

count = 0
with torch.no_grad():
    for image, label in test_loader:

        if use_cuda:
            image = image.cuda()
            label = label.cuda()
            
        y = model(image)

        pred = torch.argmax(y, dim=1)
        count += torch.sum(pred == label)

print("test accuracy: {}".format(count.item() / 10000.))
