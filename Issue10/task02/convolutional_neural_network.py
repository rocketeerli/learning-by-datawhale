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


def conv_simple(X, out_channel=3, kernel_size=(3, 5), stride=1, padding=(1, 2)):
    ''' 卷积层简洁实现 '''
    logging.info(f'shape of X: {X.shape}')
    num, in_channel, width, height = X.shape
    conv2d = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                       kernel_size=kernel_size, stride=stride, padding=padding)
    Y = conv2d(X)
    logging.info(f'shape of Y: {Y.shape}')
    logging.info(f'shape of weight: {conv2d.weight.shape}')
    logging.info(f'shape of bias: {conv2d.bias.shape}')


def pool_simple(X, kernel_size=3, padding=1, stride=(2, 1)):
    ''' 池化层简洁实现 '''
    logging.info(f'X: {X}')
    logging.info(f'shape of X: {X.shape}')
    pool2d = nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride)
    Y = pool2d(X)
    logging.info(f'Y: {Y}')
    logging.info(f'shape of Y: {Y.shape}')


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

    logging.info(f'卷积层简洁实现...')
    X = torch.rand(4, 2, 3, 5)
    conv_simple(X)

    logging.info(f'池化层简洁实现...')
    X = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)
    pool_simple(X)
