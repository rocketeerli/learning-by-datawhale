# -*- coding:utf-8 -*-
import torch
from torch import nn
import numpy as np
from data_fashion_mnist import Dataset
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s')


class MLPOriginal():
    ''' 从零开始实现的 MLP '''
    def __init__(self, dataset: Dataset, inputs=784, outputs=10, hiddens=256,
                 epochs=5, lr=0.1):
        self.dataset = dataset
        self.inputs = inputs
        self.outputs = outputs
        self.hiddens = hiddens
        self.epochs = epochs
        self.lr = lr
        self.params = self._init_coef()

    def _init_coef(self):
        W1 = torch.tensor(np.random.normal(0, 0.01, (self.inputs, self.hiddens)), dtype=torch.float32)
        b1 = torch.zeros(self.hiddens, dtype=torch.float)
        W2 = torch.tensor(np.random.normal(0, 0.01, (self.hiddens, self.outputs)), dtype=torch.float)
        b2 = torch.zeros(self.outputs, dtype=torch.float)
        params = [W1, b1, W2, b2]
        for param in params:
            param.requires_grad_(requires_grad=True)
        return params

    def relu(self, X):
        return torch.max(input=X, other=torch.tensor(0.0))

    def net(self, X):
        X = X.view((-1, self.inputs))
        H = self.relu(torch.mm(X, self.params[0]) + self.params[1])
        return torch.mm(H, self.params[2]) + self.params[3]

    def cross_entropy_loss(self, y_hat, y):
        ''' 交叉熵损失 '''
        # 先做 softmax 操作
        y_hat_exp = y_hat.exp()
        partition = y_hat_exp.sum(dim=1, keepdim=True)
        y_hat_softmax = y_hat_exp / partition
        return - torch.log(y_hat_softmax.gather(dim=1, index=y.view(-1, 1)))

    def cal_acc(self, data_iter):
        ''' 计算模型在数据集上的准确率 '''
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            acc_sum += (self.net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n

    def sgd(self):
        ''' 优化器 '''
        for param in self.params:
            param.data -= self.lr * param.grad / self.dataset.batch_size

    def grad_zero(self):
        ''' 梯度清零 '''
        for param in self.params:
            param.grad.data.zero_()

    def train(self):
        for epoch in range(self.epochs):
            loss_sum, acc_sum, n = 0.0, 0.0, 0
            for X, y in self.dataset.train_iter:
                y_hat = self.net(X)
                loss = self.cross_entropy_loss(y_hat, y).sum()
                loss.backward()
                self.sgd()
                self.grad_zero()
                loss_sum += loss.item()
                acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
                n += y.shape[0]
            with torch.no_grad():
                acc_test = self.cal_acc(self.dataset.test_iter)
            logging.info(f'epoch {epoch+1} \t train epoch loss: {loss_sum/n} \t ' +
                         f'train epoch acc: {acc_sum/n} \t test acc: {acc_test}')


class MLPPytorch():
    ''' 使用 pytorch 实现的 MLP '''
    def __init__(self, dataset: Dataset, inputs=784, outputs=10, hiddens=256,
                 epochs=5, lr=0.1):
        self.dataset = dataset
        self.inputs = inputs
        self.epochs = epochs
        self.net = nn.Sequential(
            nn.Linear(inputs, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, outputs),
        )
        for params in self.net.parameters():
            nn.init.normal_(params, mean=0, std=0.01)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

    def cal_acc(self, data_iter):
        ''' 计算模型在数据集上的准确率 '''
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            X = X.view(-1, self.inputs)
            acc_sum += (self.net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n

    def train(self):
        for epoch in range(self.epochs):
            loss_sum, acc_sum, n = 0.0, 0.0, 0
            for X, y in self.dataset.train_iter:
                X = X.view((-1, self.inputs))
                y_hat = self.net(X)
                loss = self.loss(y_hat, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item() * y.shape[0]
                acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
                n += y.shape[0]
            with torch.no_grad():
                acc_test = self.cal_acc(self.dataset.test_iter)
            logging.info(f'epoch {epoch+1} \t train epoch loss: {loss_sum/n} \t ' +
                         f'train epoch acc: {acc_sum/n} \t test acc: {acc_test}')


if __name__ == '__main__':
    batch_size = 256
    dataset = Dataset(batch_size)
    logging.info('从零开始实现 MLP...')
    MLPOriginal(dataset).train()
    logging.info('使用 PyTorch 实现的简洁版 MLP...')
    MLPPytorch(dataset).train()
