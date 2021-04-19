#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import SGD
import glob
from torch.utils.data import random_split, Subset


# In[ ]:


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
#         self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(in_features=16384, out_features=num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(ResidualBlock)


# In[ ]:


classes = ['f','j','k','l','m','n','o','x','y','z']

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
    transforms.Normalize([0.5,0.5,0.5],
                        [0.5,0.5,0.5])
])

# TODO remove color (grey scale)
# TODO remove background maybe?

train_path = 'data_letters/train'
test_path = 'data_letters/test'

train_data = torchvision.datasets.ImageFolder(train_path, transform=transformer)

train_ratio = 0.8
train_size = int(train_ratio * len(train_data))
val_size = len(train_data) - train_size
train_set, val_set = random_split(train_data, [train_size, val_size])

trainloader = DataLoader(
    train_set,
    batch_size = 256,
    shuffle = True
)

valloader = DataLoader(
    val_set,
    batch_size = 256,
    shuffle = True
)


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# New training
model = ResNet18()

# Continue training - load previous model
best_model = torch.load('model/task1-own.model')
model.load_state_dict(best_model, strict=False)

model.to(device)

# TODO: Optimizers SGD vs. Adam
# optimizer = Adam(model.parameters(), lr=0.001)
optimizer = SGD(model.parameters(), lr=0.001)
# optimizer = SGD(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_function = nn.CrossEntropyLoss()

num_epoches = 20

train_count = len(train_set)
val_count = len(val_set)

best_accuracy = 86

hist = []

for epoch in range(num_epoches):
    model.to(device)
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        if torch.cuda.is_available():
            images = images.to(device)
            labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data,1)
        train_accuracy += int(torch.sum(prediction == labels.data))
    
    train_accuracy = train_accuracy / train_count
    train_loss = train_loss/train_count
    print(f"Epoch {epoch}")
    print(f"Train_accuracy {train_accuracy}")
    
    model.eval()
    val_accuracy = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(valloader):
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)
            outputs = model(images)
            _,prediction = torch.max(outputs.data,1)
            val_accuracy += int(torch.sum(prediction==labels.data))

        val_accuracy = val_accuracy / val_count
        print(f"Train Loss {train_loss}")
        print(f"Val Accuracy {val_accuracy}")
        
        hist.append(val_accuracy)
        if val_accuracy > best_accuracy:
            print(f"Best Val Accuracy {val_accuracy}")
            torch.save(model.state_dict(), 'model/task1-own.model')
            best_accuracy = val_accuracy


# In[ ]:




