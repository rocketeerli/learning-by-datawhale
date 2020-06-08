# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s')


def data_generator(test_num=100, train_num=100, w=[1.2, -3.4, 5.6], b=5):
    features = torch.randn((train_num + test_num, 1))
    poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
    labels = (w[0] * poly_features[:, 0] + w[1] * poly_features[:, 1]
                       + w[2] * poly_features[:, 2] + b)
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


if __name__ == '__main__':
    features, labels = data_generator()
    train_data = (features[:100], labels[:100])
    test_data = (features[100:], labels[100:])
    MultivariateFunctionFitting(train_data, test_data).train()
