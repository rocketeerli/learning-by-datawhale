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


def data_generator(inputs=2, num=1000, w=[2, -4.13], b=2.6):
    ''' 数据生成器，生成需要进行拟合的数据
    :param inputs: 生成数据的维度
    :param num: 生成数据的数目
    :param w: 数据真正的系数矩阵
    :param b: 数据真正的偏移
    :return: 生成的数据 特征 和 标签
    '''
    features = torch.randn(num, inputs, dtype=torch.float32)  # 从正态分布中获取随机数
    labels = w[0] * features[:, 0] + w[1] * features[:, 1] + b  # 生成标准数据
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                           dtype=torch.float32)  # 添加噪声
    # 展示数据
    # plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
    # plt.show()
    return features, labels


def data_loader(features, labels, batch_size=10):
    ''' 加载数据 
    :param featues: 样本的特征矩阵
    :param labels: 样本的标签集合
    :param batch_size: batch 大小
    :return: 每个 batch 的 数据和标签
    '''
    num = len(features)
    indices = list(range(num))
    random.shuffle(indices)  # 打乱数据顺序
    for i in range(0, num, batch_size):
        # 构建下标 tensor, 按行进行索引
        j = torch.LongTensor(indices[i: min(i + batch_size, num)])
        yield features.index_select(0, j), labels.index_select(0, j)


def init_coef():
    ''' 初始化模型参数 '''
    w = torch.tensor(np.random.normal(0, 0.01, (NUM_IMPUTS, 1)), dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)
    w.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)
    return w, b


def linreg(X, w, b):
    ''' 定义线性回归模型 '''
    # 矩阵相乘
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    ''' 定义均方误差损失函数
    :param y_hat: 预测值
    :param y: 真实值
    :return: loss 值
    '''
    return ((y_hat - y.view(y_hat.size())) ** 2).sum() / 2


def sgd(params, lr, batch_size=10):
    ''' 定义优化函数
    :param params: 参数
    :param lr: 学习率
    :param batch_size: batch 大小
    '''
    for param in params:
        param.data -= lr * param.grad / batch_size


def train(features, labels):
    lr = 0.03
    num_epochs = 5

    w, b = init_coef()

    # 开始训练
    for epoch in range(num_epochs):
        for X, y in data_loader(features, labels):
            loss = squared_loss(linreg(X, w, b), y)
            loss.backward()
            sgd([w, b], lr)
            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = squared_loss(linreg(features, w, b), labels)
        logging.info('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))


if __name__ == '__main__':
    features, labels = data_generator(NUM_IMPUTS)
    train(features, labels)

