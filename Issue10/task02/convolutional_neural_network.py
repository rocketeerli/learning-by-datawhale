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


if __name__ == '__main__':
    X = torch.tensor(np.arange(9)).view(3, 3)
    Kernal = torch.tensor(np.arange(4)).view(2, 2)
    Y = corr2d(X, Kernal)
    logging.info(f'\nY: {Y}')
