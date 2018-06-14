from __future__ import print_function, division

import os
import shutil
import torch
import collections
from torchvision import transforms,datasets

import os
import torch
import pylab
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
from PIL import Image



plt.ion()  # interactive mode

data_transform = transforms.Compose([
    transforms.Resize(84),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root='train/', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = datasets.ImageFolder(root='valid/', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 18 * 18, 800)
        self.fc2 = nn.Linear(800, 120)
        self.fc3 = nn.Linear(120, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


import torch.optim as optim
import datetime

net = Net()
cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

for epoch in range(3):
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = cirterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if i % 2000 == 1999:
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
            print('[%d %5d] loss: %.3f, Time:' % (epoch + 1, i + 1, running_loss / 2000))
            print(nowTime)
            running_loss = 0.0

    print('finished training!')

    correct = 0
    total = 0

    for data in test_loader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 5000 test images: %d %%' % (100 * correct / total))