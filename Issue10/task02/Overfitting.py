# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from data_fashion_mnist import Dataset
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
                loss = self.loss(self.net(X), y)
                self.optimizer_w.zero_grad()
                self.optimizer_b.zero_grad()
                loss.backward()
                self.optimizer_w.step()
                self.optimizer_b.step()
            train_loss = self.loss(self.net(self.train_features), self.train_labels).item()
            test_loss = self.loss(self.net(self.test_features), self.test_labels).item()
            train_ls.append(train_loss)
            test_ls.append(test_loss)
            logging.info(f'epoch {epoch}: train loss: {train_loss}' +
                         f'test loss: {test_loss}')
        logging.info(f'L2 norm of w: {self.net.weight.data.norm().item()}')
        show_data(range(1, epochs+1), train_ls, 'epochs', 'loss',
                  range(1, epochs+1), test_ls, ['train', 'test'])
        plt.show()


class DropoutOriginal():
    ''' 利用四层 MLP 从零开始实现 Dropout '''
    def __init__(self, coef: list, drop_prob: list):
        assert len(coef) == 4
        assert len(drop_prob) == 2
        self.coef = coef
        self.drop_prob1 = drop_prob[0]
        self.drop_prob2 = drop_prob[1]
        self.params = self._init_coef(*coef)

    def _init_coef(self, *args):
        ''' 参数的初始化 '''
        inputs, hiddens1, hiddens2, outputs = args
        W1 = torch.tensor(np.random.normal(0, 0.01, size=(inputs, hiddens1)),
                          dtype=torch.float, requires_grad=True)
        b1 = torch.zeros(hiddens1, requires_grad=True)
        W2 = torch.tensor(np.random.normal(0, 0.01, size=(hiddens1, hiddens2)),
                          dtype=torch.float, requires_grad=True)
        b2 = torch.zeros(hiddens2, requires_grad=True)
        W3 = torch.tensor(np.random.normal(0, 0.01, size=(hiddens2, outputs)),
                          dtype=torch.float, requires_grad=True)
        b3 = torch.zeros(outputs, requires_grad=True)
        return [W1, b1, W2, b2, W3, b3]

    def net(self, X, is_training=True):
        X = X.view(-1, self.coef[0])
        H1 = (torch.matmul(X, self.params[0]) + self.params[1]).relu()
        if is_training:
            H1 = self.dropout(H1, self.drop_prob1)
        H2 = (torch.matmul(H1, self.params[2]) + self.params[3]).relu()
        if is_training:
            H2 = self.dropout(H2, self.drop_prob2)
        return torch.matmul(H2, self.params[4]) + self.params[5]

    def dropout(self, X, drop_prob):
        assert 0 <= drop_prob <= 1
        X = X.float()
        keep_prob = 1 - drop_prob
        if keep_prob == 0:
            return torch.zeros_like(X)
        mask = (torch.rand(X.shape) < keep_prob).float()
        return mask * X / keep_prob

    def evaluate_accuracy(self, data_iter):
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            X = X.view(-1, self.coef[0])
            acc_sum += (self.net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n

    def train(self, dataset: Dataset, epochs=5, lr=0.1):
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.params, lr=lr)
        for epoch in range(epochs):
            loss_sum, acc_sum, n = 0.0, 0.0, 0
            for X, y in dataset.train_iter:
                X = X.view(-1, self.coef[0])
                y_hat = self.net(X)
                ls = loss(y_hat, y)
                optimizer.zero_grad()
                ls.backward()
                optimizer.step()
                # 数据统计
                num = y.shape[0]
                n += num
                loss_sum += ls.item() * num
                acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            with torch.no_grad():
                test_acc = self.evaluate_accuracy(dataset.test_iter)
            logging.info(f'epoch {epoch+1} \t train epoch loss: {loss_sum/n} \t ' +
                         f'train epoch acc: {acc_sum/n} \t test acc: {test_acc}')


class DropoutPytorch():
    def __init__(self, coef: list, drop_prob: list, lr=0.1):
        assert len(coef) == 4 and len(drop_prob) == 2
        self.inputs = coef[0]
        self.net = nn.Sequential(
            nn.Linear(coef[0], coef[1]),
            nn.ReLU(),
            nn.Dropout(drop_prob[0]),
            nn.Linear(coef[1], coef[2]),
            nn.ReLU(),
            nn.Dropout(drop_prob[1]),
            nn.Linear(coef[2], coef[3]),
        )
        for param in self.net.parameters():
            nn.init.normal_(param, mean=0, std=0.01)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

    def evaluate_accuracy(self, data_iter):
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            X = X.view(-1, self.inputs)
            self.net.eval()
            acc_sum += (self.net(X).argmax(dim=1) == y).float().sum().item()
            self.net.train()
            n += y.shape[0]
        return acc_sum / n

    def train(self, dataset: Dataset, epochs=5):
        for epoch in range(epochs):
            loss_sum, acc_sum, n = 0.0, 0.0, 0
            for X, y in dataset.train_iter:
                X = X.view(-1, self.inputs)
                y_hat = self.net(X)
                loss = self.loss(y_hat, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 数据统计
                num = y.shape[0]
                n += num
                loss_sum += loss.item() * num
                acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            with torch.no_grad():
                test_acc = self.evaluate_accuracy(dataset.test_iter)
            logging.info(f'epoch {epoch+1} \t train epoch loss: {loss_sum/n} \t ' +
                         f'train epoch acc: {acc_sum/n} \t test acc: {test_acc}')


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

    logging.info(f'丢弃法 从零开始实现的 MLP')
    batch_size = 256
    dataset = Dataset(batch_size)
    neur_nums = [784, 256, 256, 10]
    drop_prob = [0.2, 0.5]
    DropoutOriginal(neur_nums, drop_prob).train(dataset)
    logging.info(f'丢弃法 MLP简洁版')
    DropoutPytorch(neur_nums, drop_prob).train(dataset)
