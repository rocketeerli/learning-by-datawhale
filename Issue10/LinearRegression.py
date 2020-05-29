# -*- coding:utf-8  -*-
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s')

NUM_IMPUTS = 2  # 数据的维度
BATCH_SIZE = 10


class Dataset():
    ''' 数据集操作 
    生成需要进行拟合的数据; 并加载数据 
    :param inputs: 生成数据的维度
    :param num: 生成数据的数目
    :param w: 数据真正的系数矩阵
    :param b: 数据真正的偏移
    '''
    def __init__(self, inputs=2, num=1000, w=[2, -4.13], b=2.6):
        self.inputs = inputs
        self.num = num
        self.w = w
        self.b = b
        self.features, self.labels = self._data_generator()

    def _data_generator(self):
        features = torch.randn(self.num, self.inputs, dtype=torch.float32)  # 从正态分布中获取随机数
        labels = self.w[0] * features[:, 0] + self.w[1] * features[:, 1] + self.b  # 生成标准数据
        labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                               dtype=torch.float32)  # 添加噪声
        return features, labels

    def data_loader(self, batch_size=10):
        ''' 数据加载器
        :param batch_size: batch 大小
        :return: 每个 batch 的 数据和标签
        '''
        num = len(self.features)
        indices = list(range(num))
        random.shuffle(indices)  # 打乱数据顺序
        for i in range(0, num, batch_size):
            # 构建下标 tensor, 按行进行索引
            j = torch.LongTensor(indices[i: min(i + batch_size, num)])
            yield self.features.index_select(0, j), self.labels.index_select(0, j)


class LinerRegOriginal():
    ''' 从零开始做线性回归 '''
    def __init__(self, lr=0.03, epochs=5, batch_size=10):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.w, self.b = self.init_coef()

    def init_coef(self):
        ''' 初始化模型参数 '''
        w = torch.tensor(np.random.normal(0, 0.01, (NUM_IMPUTS, 1)), dtype=torch.float32)
        b = torch.zeros(1, dtype=torch.float32)
        w.requires_grad_(requires_grad=True)
        b.requires_grad_(requires_grad=True)
        return w, b

    def linreg(self, X):
        ''' 定义线性回归模型 '''
        # 矩阵相乘
        return torch.mm(X, self.w) + self.b

    def squared_loss(self, y_hat, y):
        ''' 定义均方误差损失函数
        :param y_hat: 预测值
        :param y: 真实值
        :return: loss 值
        '''
        return ((y_hat - y.view(y_hat.size())) ** 2).sum() / 2

    def sgd(self):
        ''' 定义优化函数 '''
        for param in [self.w, self.b]:
            param.data -= self.lr * param.grad / self.batch_size

    def train(self, dataset):
        features, labels = dataset.features, dataset.labels
        # 开始训练
        for epoch in range(self.epochs):
            for X, y in dataset.data_loader():
                loss = self.squared_loss(self.linreg(X), y)
                loss.backward()
                self.sgd()
                self.w.grad.data.zero_()
                self.b.grad.data.zero_()
            with torch.no_grad():
                train_loss = self.squared_loss(self.linreg(features), labels)
            logging.info('epoch %d, loss %f' % (epoch + 1, train_loss.item()))


if __name__ == '__main__':
    dataset = Dataset()
    LinerRegOriginal().train(dataset)
