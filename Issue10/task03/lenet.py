# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


class Flatten(nn.Module):
    ''' 展平操作 '''
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(nn.Module):
    ''' 将图像大小重定型 '''
    def forward(self, x):
        return x.view(-1, 1, 28, 28)  # (B x C x H x W)


class LeNet():
    def __init__(self):
        self.net = nn.Sequential(
            Reshape(),
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )


def test(X=torch.randn(size=(1, 1, 28, 28), dtype=torch.float32)):
    le = LeNet()
    for layer in le.net:
        X = layer(X)
        logging.info(f"{layer.__class__.__name__} output shape: \t {X.shape}")


if __name__ == '__main__':
    test()
