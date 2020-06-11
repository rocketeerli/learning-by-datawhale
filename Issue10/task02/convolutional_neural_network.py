# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format= '%(asctime)s - %(levelname)s: %(message)s')


def corr2d(X, Kernal):
    ''' 卷积 二维互相关运算 '''
    H, W = X.shape
    h, w = Kernal.shape
    Y = torch.zeros(H - h + 1, W - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * Kernal).sum()
    return Y


class Conv2D(nn.Module):
    ''' 二维卷积层 '''
    def __init__(self, kernal_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernal_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def train_Conv2D(data, labels, kernal_size=(1, 2), epochs=30, lr=0.01):
    conv2d = Conv2D(kernal_size)
    for epoch in range(epochs):
        y_hat = conv2d(data)
        loss = ((y_hat - labels) ** 2).sum()
        loss.backward()
        # 梯度下降
        conv2d.weight.data -= lr * conv2d.weight.grad
        conv2d.bias.data -= lr * conv2d.bias.grad
        # 梯度清零
        conv2d.weight.grad.zero_()
        conv2d.bias.grad.zero_()
        logging.info(f'epoch {epoch+1}:\t loss: {loss.item()}')
    logging.info(f'weight: {conv2d.weight.data}')
    logging.info(f'bias: {conv2d.bias.data}')


if __name__ == '__main__':
    logging.info('卷积-----二维互相关运算...')
    X = torch.tensor(np.arange(9)).view(3, 3)
    Kernal = torch.tensor(np.arange(4)).view(2, 2)
    Y = corr2d(X, Kernal)
    logging.info(f'\nY: {Y}')
    logging.info('二维卷积层训练 锐化 获取图像边缘信息')
    train_data, labels = torch.ones(6, 8), torch.zeros(6, 7)
    train_data[:, 2:6] = 0
    labels[:, 1] = 1
    labels[:, 5] = -1
    logging.info(f'train data: {train_data}')
    logging.info(f'labels: {labels}')
    train_Conv2D(train_data, labels)
