# -*- coding: utf-8 -*-
import torch
import time
from torch import nn, optim
import torchvision
import numpy as np
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 第二层卷积
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 第三层卷积
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            # 第四层卷积
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            # 第五层卷积
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            # 第一层全连接
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 第二层全连接
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 最后一层全连接，输出层
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


if __name__ == '__main__':
    logging.info(device)
    net = AlexNet().to(device)
    logging.info(net)
