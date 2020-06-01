# -*- coding:utf-8  -*-
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s')


class Dataset():
    def __init__(self, batch_size=256, num_workers=4):
        self.classes = [
            't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
            'shirt', 'sneaker', 'bag', 'ankle boot'
        ]
        self.batch_size = batch_size
        self.mnist_train = torchvision.datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor())
        self.mnist_test = torchvision.datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.ToTensor())
        self.train_iter = DataLoader(self.mnist_train,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)
        self.test_iter = DataLoader(self.mnist_test,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        logging.info(type(self.mnist_train))
        logging.info(f'mnist train length: {len(self.mnist_train)}')
        logging.info(f'mnist test length: {len(self.mnist_test)}')

    def get_labels(self, labels):
        return [self.classes[int(i)] for i in labels]

    def show_data(self, images, labels):
        labels = self.get_labels(labels)
        _, figs = plt.subplots(1, len(images), figsize=(12, 12))
        for f, img, lbl in zip(figs, images, labels):
            f.imshow(img.view((28, 28)).numpy())
            f.set_title(lbl)
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)
        plt.show()


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
            logging.info(f'epoch {epoch+1} \t loss {train_loss_sum} \t ' +
                         f'train acc {train_acc_sum/n} \t test acc {test_acc}')


def test():
    dataset = Dataset()
    # 显示数据
    X, y = [], []
    for i in range(10):
        X.append(dataset.mnist_train[i][0])
        y.append(dataset.mnist_train[i][1])
    dataset.show_data(X, y)
    # 计算加载全部数据的时间
    start = time.time()
    for X, y in dataset.train_iter:
        continue
    logging.info('%.2f sec' % (time.time() - start))
    # 
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    logging.info(f'sum dim 0 keepdim True {X.sum(dim=0, keepdim=True)}')
    logging.info(f'sum dim 1 keepdim True {X.sum(dim=1, keepdim=True)}')
    logging.info(f'sum dim 0 keepdim False {X.sum(dim=0, keepdim=False)}')
    logging.info(f'sum dim 1 keepdim False {X.sum(dim=1, keepdim=False)}')


if __name__ == '__main__':
    dataset = Dataset()
    model = SoftmaxOriginal(dataset)
    model.train()
