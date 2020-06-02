# -*- coding:utf-8  -*-
from data_fashion_mnist import Dataset
import torch
import numpy as np
from torch import nn
from torch.nn import init
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s')


class SoftmaxOriginal():
    ''' 从零开始做 Softmax '''
    def __init__(self, dataset, inputs=784, outputs=10, lr=0.1, epochs=5):
        self.dataset = dataset
        self.inputs = inputs
        self.outputs = outputs
        self.lr = lr
        self.epochs = epochs
        self.W, self.b = self._init_coef()

    def _init_coef(self):
        W = torch.tensor(np.random.normal(0, 0.01,
                                          (self.inputs, self.outputs)),
                         dtype=torch.float)
        b = torch.zeros(self.outputs, dtype=torch.float)
        W.requires_grad_(requires_grad=True)
        b.requires_grad_(requires_grad=True)
        return W, b

    def softmax(self, X):
        X_exp = X.exp()
        partition = X_exp.sum(dim=1, keepdim=True)
        return X_exp / partition

    def net(self, X):
        ''' softmax 回归模型 '''
        return self.softmax(torch.mm(X.view((-1, self.inputs)), self.W) + self.b)

    def cross_entropy(self, y_hat, y):
        ''' 交叉熵损失函数
        :param y_hat: n x outputs
        :param y: n
        '''
        return - torch.log(y_hat.gather(1, y.view(-1, 1)))

    def sgd(self):
        ''' 定义优化函数 '''
        for param in [self.W, self.b]:
            param.data -= self.lr * param.grad / self.dataset.batch_size

    def evaluate_accuracy(self, data_iter):
        ''' 计算模型准确率 '''
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            acc_sum += (self.net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n

    def train(self):
        for epoch in range(self.epochs):
            train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
            for X, y in self.dataset.train_iter:
                y_hat = self.net(X)
                loss = self.cross_entropy(y_hat, y).sum()
                loss.backward()
                self.sgd()
                self.W.grad.data.zero_()
                self.b.grad.data.zero_()
                # 数据收集
                train_loss_sum += loss.item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                n += y.shape[0]
            with torch.no_grad():
                test_acc = self.evaluate_accuracy(self.dataset.test_iter)
            logging.info(f'epoch {epoch+1} \t loss {train_loss_sum/n} \t ' +
                         f'train acc {train_acc_sum/n} \t test acc {test_acc}')


class LinearNet(nn.Module):
    ''' 线性分类模型 '''
    def __init__(self, inputs=784, outputs=10):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(inputs, outputs)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y


class SoftmaxPytorch():
    def __init__(self, dataset, inputs=784, outputs=10, lr=0.1, epochs=5):
        self.dataset = dataset
        self.epochs = epochs
        self.net = LinearNet(inputs=784, outputs=10)
        # 初始化模型参数
        init.normal_(self.net.linear.weight, mean=0, std=0.01)
        init.constant_(self.net.linear.bias, val=0)
        # 定义损失函数
        self.loss = nn.CrossEntropyLoss()
        # 定义优化函数
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

    def evaluate_accuracy(self, data_iter):
        ''' 计算模型准确率 '''
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            acc_sum += (self.net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n

    def train(self):
        for epoch in range(self.epochs):
            train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
            for X, y in self.dataset.train_iter:
                y_hat = self.net(X)
                loss = self.loss(y_hat, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 数据统计
                train_loss_sum += loss.item() * y.shape[0]
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                n += y.shape[0]
            with torch.no_grad():
                test_acc = self.evaluate_accuracy(self.dataset.test_iter)
            logging.info(f'epoch {epoch+1} \t loss {train_loss_sum/n} \t ' +
                         f'train acc {train_acc_sum/n} \t test acc {test_acc}')


if __name__ == '__main__':
    dataset = Dataset()
    logging.info('从零开始做 Softmax...')
    model = SoftmaxOriginal(dataset)
    model.train()
    logging.info('使用 Pytorch 的简洁实现版...')
    model_pytorch = SoftmaxPytorch(dataset)
    model_pytorch.train()
