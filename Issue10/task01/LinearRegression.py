# -*- coding:utf-8  -*-
import torch
from torch import nn
import numpy as np
import random
import logging
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s')
torch.manual_seed(1)  # 固定 pytorch 的随机数
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

    def data_loader_torch(self, batch_size=10):
        ''' 利用 torch 加载数据集 '''
        dataset = Data.TensorDataset(self.features, self.labels)
        data_iter = Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        return data_iter


class LinearRegressionOriginal():
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
        return ((y_hat - y.view(y_hat.size())) ** 2 / 2).mean()

    def sgd(self):
        ''' 定义优化函数 '''
        for param in [self.w, self.b]:
            param.data -= self.lr * param.grad / self.batch_size

    def train(self, dataset):
        features, labels = dataset.features, dataset.labels
        # 开始训练
        for epoch in range(self.epochs):
            for (X, y) in dataset.data_loader():
                loss = self.squared_loss(self.linreg(X), y)
                loss.backward()
                self.sgd()
                self.w.grad.data.zero_()
                self.b.grad.data.zero_()
            with torch.no_grad():
                train_loss = self.squared_loss(self.linreg(features), labels)
            logging.info('epoch %d, loss %f' % (epoch + 1, train_loss.item()))


class LinearNet(nn.Module):
    ''' 定义线性回归模型 '''
    def __init__(self, n_feature=2):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


class LinearRegressionPytorch():
    ''' 使用 pytorch 实现的线性分类器 '''
    def __init__(self, num_inputs=2, lr=0.03, epochs=5):
        self.epochs = epochs
        self.net = LinearNet(num_inputs)
        self.init_coef()
        self.loss = nn.MSELoss()  # 定义损失函数
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)  # 定义优化函数

    def init_coef(self):
        ''' 初始化模型参数 '''
        init.normal_(self.net.linear.weight, mean=0.0, std=0.01)
        init.constant_(self.net.linear.bias, val=0.0)

    def train(self, dataset):
        for epoch in range(1, self.epochs + 1):
            for X, y in dataset.data_loader_torch():
                output = self.net(X)
                loss = self.loss(output, y.view(-1, 1))
                self.optimizer.zero_grad()  # 重置梯度
                loss.backward()
                self.optimizer.step()
            logging.info('epoch %d, loss: %f' % (epoch, loss.item()))


if __name__ == '__main__':
    dataset = Dataset()
    logging.info(f'从零开始的线性回归')
    LinearRegressionOriginal().train(dataset)
    logging.info(f'使用 PyTorch 的线性回归')
    LinearRegressionPytorch().train(dataset)
