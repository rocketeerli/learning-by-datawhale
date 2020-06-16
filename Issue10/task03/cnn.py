# -*- coding: utf-8 -*-
import torch
import time
from torch import nn, optim
import torchvision
import numpy as np
import torch.nn.functional as F
from data_fashion_mnist import Dataset
from lenet import evaluate_accuracy
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr, epochs, batch_size = 0.001, 3, 16


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


def train(net: nn.Module, dataset: Dataset, criterion, optimizer, lr=None):
    for epoch in range(epochs):
        train_ls = torch.tensor([0], dtype=torch.float32, device=device)
        train_acc = torch.tensor([0], dtype=torch.float32, device=device)
        n, start = 0, time.time()
        for X, y in dataset.train_iter:
            X, y = X.to(device), y.to(device)
            net.train()
            optimizer.zero_grad()
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y = y.long()
                train_ls += loss.float() * y.shape[0]
                train_acc += torch.sum(torch.argmax(y_hat, dim=1) == y).float()
                n += y.shape[0]
        test_acc = evaluate_accuracy(dataset.test_iter, net)
        logging.info(f'epoch {epoch+1}: \t loss: {train_ls.item()/n} \t train acc:' +
                     f'{train_acc.item()/n} \t test acc: {test_acc} \t time: {time.time()-start}')

if __name__ == '__main__':
    logging.info(f'running on {device} ...')
    dataset = Dataset(batch_size, resize=224)

    logging.info('AlexNet ...')
    net = AlexNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train(net, dataset, criterion, optimizer)
