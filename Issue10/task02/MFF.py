# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s')


def data_generator(num=200, inputs=1, b=5, w=[1.2, -3.4, 5.6]):
    ''' 数据生成器
    如果是一维，则生成一维的三阶多项式数据
    如果是高维，则生成多维数据
    '''
    features = torch.randn((num, inputs))
    if inputs == 1:
        poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
        labels = (w[0] * poly_features[:, 0] + w[1] * poly_features[:, 1]
                  + w[2] * poly_features[:, 2] + b)
    else:
        w = torch.ones((inputs, 1)) * 0.01
        labels = torch.matmul(features, w) + b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
    return features, labels


def show_data(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
              legend=None, figsize=(3.5, 2.5)):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)  # y 轴使用对数尺度
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


class MultivariateFunctionFitting():
    def __init__(self, train_data, test_data, epochs=100, loss=None):
        # 初始化数据
        self.train_features = train_data[0]
        self.train_labels = train_data[1]
        self.test_features = test_data[0]
        self.test_labels = test_data[1]
        self.epochs = epochs
        self.loss = loss if loss else torch.nn.MSELoss()
        # 初始化网络模型
        self.net = torch.nn.Linear(self.train_features.shape[-1], 1)
        # 设置批量大小
        self.batch_size = min(10, self.train_labels.shape[0])
        self.dataset = Data.TensorDataset(self.train_features, self.train_labels)
        self.train_iter = Data.DataLoader(self.dataset, self.batch_size, shuffle=True)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)

    def train(self):
        train_loss, test_loss = [], []
        for epoch in range(self.epochs):
            for X, y in self.train_iter:
                loss = self.loss(self.net(X), y.view(-1, 1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_labels = self.train_labels.view(-1, 1)
            test_labels = self.test_labels.view(-1, 1)
            train_epoch_ls = self.loss(self.net(self.train_features), train_labels).item()
            test_epoch_ls = self.loss(self.net(self.test_features), test_labels).item()
            logging.info(f'epoch {epoch} \t train loss: {train_epoch_ls} \t' +
                         f'test loss: {test_epoch_ls}')
            train_loss.append(train_epoch_ls)
            test_loss.append(test_epoch_ls)
        logging.info(f'final epoch: train loss: {train_loss[-1]}\t test loss: {test_loss[-1]}')
        show_data(range(1, self.epochs+1), train_loss, 'epochs', 'loss',
                  range(1, self.epochs+1), test_loss, ['train', 'test'])
        logging.info(f'weight: {self.net.weight.data} \n bias: {self.net.bias.data}')


class MultiLinearRegression():
    def __init__(self, train_data, test_data):
        self.train_features = train_data[0]
        self.train_labels = train_data[1]
        self.test_features = test_data[0]
        self.test_labels = test_data[1]
        self.inputs = self.train_features.shape[-1]
        self.params = self.init_params()

    def init_params(self):
        ''' 初始化线性回归的参数 '''
        w = torch.randn((self.inputs, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        return [w, b]

    def linreg(self, X):
        ''' 定义线性回归模型 '''
        return torch.mm(X, self.params[0]) + self.params[1]

    def square_loss(self, y_hat, y):
        ''' 均方误差 '''
        return (y_hat - y.view(y_hat.size())) ** 2 / 2

    def l2_penalty(self, w):
        ''' L2 惩罚项 '''
        return (w**2).sum() / 2

    def sgd(self, batch_size, lr):
        ''' 定义优化函数 '''
        for param in self.params:
            param.data -= lr * param.grad / batch_size

    def train(self, epochs=100, batch_size=1, lr=0.003, lambd=0.1):
        dataset = torch.utils.data.TensorDataset(self.train_features, self.train_labels)
        train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
        train_ls, test_ls = [], []
        for epoch in range(epochs):
            for X, y in train_iter:
                loss = self.square_loss(self.linreg(X), y) + lambd * self.l2_penalty(self.params[0])
                loss = loss.sum()
                if self.params[0].grad is not None:
                    self.params[0].grad.data.zero_()
                    self.params[1].grad.data.zero_()
                loss.backward()
                self.sgd(batch_size, lr)
            with torch.no_grad():
                train_loss = (self.square_loss(self.linreg(self.train_features), self.train_labels)
                              + self.l2_penalty(self.params[0])).mean().item()
                test_loss = (self.square_loss(self.linreg(self.test_features), self.test_labels)
                             + self.l2_penalty(self.params[1])).mean().item()
                logging.info(f'epoch {epoch}: train loss: {train_loss}' +
                             f'test loss: {test_loss}')
            train_ls.append(train_loss)
            test_ls.append(test_loss)
        show_data(range(1, epochs+1), train_ls, 'epochs', 'loss',
                  range(1, epochs+1), test_ls, ['train', 'test'])
        logging.info(f'L2 norm of w: {self.params[0].norm().item()}')
        plt.show()


class MultiLRSimple():
    ''' 高维线性回归 简洁实现 '''
    def __init__(self, train_data, test_data, lr=0.003, wd=3):
        self.train_features = train_data[0]
        self.train_labels = train_data[1]
        self.test_features = test_data[0]
        self.test_labels = test_data[1]
        # 初始化模型
        inputs = self.train_features.shape[-1]
        self.net = nn.Linear(inputs, 1)
        nn.init.normal_(self.net.weight, mean=0, std=1)
        nn.init.normal_(self.net.bias, mean=0, std=1)
        self.loss = nn.MSELoss()
        self.optimizer_w = torch.optim.SGD(params=[self.net.weight], lr=lr, weight_decay=wd)
        self.optimizer_b = torch.optim.SGD(params=[self.net.bias], lr=lr)

    def train(self, epochs=100, batchsize=1):
        dataset = torch.utils.data.TensorDataset(self.train_features, self.train_labels)
        train_iter = torch.utils.data.DataLoader(dataset, batchsize, shuffle=True)
        train_ls, test_ls = [], []
        for epoch in range(epochs):
            for X, y in train_iter:
                loss = self.loss(self.net(X), y).mean()
                self.optimizer_w.zero_grad()
                self.optimizer_b.zero_grad()
                loss.backward()
                self.optimizer_w.step()
                self.optimizer_b.step()
            train_loss = self.loss(self.net(self.train_features), self.train_labels).mean().item()
            test_loss = self.loss(self.net(self.test_features), self.test_labels).mean().item()
            train_ls.append(train_loss)
            test_ls.append(test_loss)
            logging.info(f'epoch {epoch}: train loss: {train_loss}' +
                         f'test loss: {test_loss}')
        logging.info(f'L2 norm of w: {self.net.weight.data.norm().item()}')
        show_data(range(1, epochs+1), train_ls, 'epochs', 'loss',
                  range(1, epochs+1), test_ls, ['train', 'test'])
        plt.show()


if __name__ == '__main__':
    logging.info(f'多项式拟合 ...')
    n_train, n_test = 100, 100
    features, labels = data_generator(num=n_train+n_test)
    train_data = (features[:n_train, :], labels[:n_train])
    test_data = (features[n_train:, :], labels[n_train:])
    MultivariateFunctionFitting(train_data, test_data).train()
    logging.info(f'高维线性回归 从零开始实现 ...')
    n_train, n_test = 100, 100
    features, labels = data_generator(n_train+n_test, inputs=200, b=0.05)
    train_data = (features[:n_train, :], labels[:n_train])
    test_data = (features[n_train:], labels[n_train:])
    MultiLinearRegression(train_data, test_data).train(lambd=3)
    logging.info(f'高维线性回归 简洁实现 ...')
    MultiLRSimple(train_data, test_data).train()
