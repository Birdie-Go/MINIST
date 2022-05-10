# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn import Sequential

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 第一层卷积层
        self.conv1 = Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二卷积层
        self.conv2 = Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 全连接层
        self.dense = Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    # 正向传播
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = x2.view(-1, 7 * 7 * 128)
        x = self.dense(x)
        return x