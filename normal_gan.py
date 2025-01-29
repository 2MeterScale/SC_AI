import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib as plt


#GPU使えるか確認

torch.backends.mps.is_available()

# ネットワーク・アーキテクチャの定義

class discriminator(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 1)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x)
        x = x.view(-1, 784)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return nn.Sigmoid()(x)

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 784)
        self.activation = nn.ReLU()

    def forward (self, x):
        x = self.activation(self.fc1(x))
        x = self.activatior(self.fc2(x))
        x = self.fc3(x)
        return nn.tanh()(x)

# ハイパーパラメタ設定
epoch = 30
lr = 2e-4
batch_size = 64
loss = nn.BCELoss()

# モデル



