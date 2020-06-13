# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
from data_fashion_mnist import Dataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def try_gpu():
    ''' 尝试使用 GPU '''
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return device


class Flatten(nn.Module):
    ''' 展平操作 '''
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(nn.Module):
    ''' 将图像大小重定型 '''
    def forward(self, x):
        return x.view(-1, 1, 28, 28)  # (B x C x H x W)


LeNet = nn.Sequential(
        Reshape(),
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        Flatten(),
        nn.Linear(in_features=16*5*5, out_features=120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10),
    )


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def evaluate_accuracy(data_iter, net):
    ''' 计算准确率 '''
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        net.eval()
        with torch.no_grad():
            y = y.long()
            acc = torch.sum((torch.argmax(net(X), dim=1) == y))
            acc_sum += acc
            n += y.shape[0]
    return acc_sum.item() / n


def train(net: nn.Module, dataset: Dataset, criterion, epochs, lr=None):
    logging.info(f'training on {device}')
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    for epoch in range(epochs):
        train_ls = torch.tensor([0], dtype=torch.float32, device=device)
        train_acc = torch.tensor([0], dtype=torch.float32, device=device)
        n, start = 0, time.time()
        for X, y in dataset.train_iter:
            X, y = X.to(device), y.to(device)
            net.train()
            optimizer.zero_grad()
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y = y.long()
                train_ls += loss.float() * y.shape[0]
                train_acc += torch.sum(torch.argmax(y_hat, dim=1) == y).float()
                n += y.shape[0]
        test_acc = evaluate_accuracy(dataset.test_iter, net)
        logging.info(f'epoch {epoch+1}: \t loss: {train_ls.item()/n} \t train acc:' +
                     f'{train_acc.item()/n} \t test acc: {test_acc} \t time: {time.time()-start}')


def show_result(net, dataset):
    for X, y in dataset.test_iter:
        X, y = X.to(device), y.to(device)
        break
    logging.info(f'X shape: {X.shape} \t y shape: {y.shape}')
    net.eval()
    y_pre = net(X)
    logging.info(f'y_pre 10: {torch.argmax(y_pre, dim=1)[:10]}')
    logging.info(f'y truth 10: {y[:10]}')


def test(X=torch.randn(size=(1, 1, 28, 28), dtype=torch.float32)):
    for layer in LeNet:
        X = layer(X)
        logging.info(f"{layer.__class__.__name__} output shape: \t {X.shape}")


if __name__ == '__main__':
    device = try_gpu()

    logging.info('test every layer of LeNet...')
    test()

    logging.info('preform LeNet on fashion mnist dataset...')
    dataset = Dataset(batch_size=256)
    train_iter, test_iter = dataset.train_iter, dataset.test_iter
    logging.info(f'lenght of trian iter: {len(train_iter)}')
    # 初始化网络
    LeNet.apply(init_weights)
    LeNet.to(device)
    # 设置训练参数
    lr, epochs = 0.9, 10
    criterion = nn.CrossEntropyLoss()
    # 开始训练
    train(LeNet, dataset, criterion, epochs, lr)
    # 测试集上可视化
    show_result(LeNet, dataset)
