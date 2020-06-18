# -*- coding:utf-8 -*-
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import logging
from lenet import Flatten
from cnn import train, GlobalAvgPool2d
from data_fashion_mnist import Dataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levalname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features) #全连接层输出神经元
        else:
            shape = (1, num_features, 1, 1)  #通道数
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = batch_norm(self.training,
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y


def batch_norm(training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
    if not training:
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
            # X的形状以便后面可以做广播运算
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var


class LeNetBatchNormalization():
    def __init__(self, simple=True):
        self.net = self.modle_simple() if simple else self.moudle_complex()

    def moudle_complex(self):
        return nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            BatchNorm(6, num_dims=4),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            BatchNorm(16, num_dims=4),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(16*4*4, 120),
            BatchNorm(120, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            BatchNorm(84, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def modle_simple(self):
        return nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(16*4*4, 120),
            nn.BatchNorm1d(120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )


def lenet_bn(lr, simple=True):
    lenet = LeNetBatchNormalization(simple=simple).net.to(device)
    optimizer = torch.optim.Adam(lenet.parameters(), lr=lr)
    train(lenet, dataset, criterion, optimizer)


class Residual(nn.Module):
    ''' 残差块 '''
    # 可以设定输出通道数、是否使用额外的1x1卷积层来修改通道数以及卷积层的步幅。
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class ResNet():
    def __init__(self):
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.net.add_module("resnet_block1", self.resnet_block(64, 64, 2, first_block=True))
        self.net.add_module("resnet_block2", self.resnet_block(64, 128, 2))
        self.net.add_module("resnet_block3", self.resnet_block(128, 256, 2))
        self.net.add_module("resnet_block4", self.resnet_block(256, 512, 2))
        self.net.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
        self.net.add_module("fc", nn.Sequential(Flatten(), nn.Linear(512, 10))) 

    def resnet_block(self, in_channels, out_channels, num_residuals, first_block=False):
        if first_block:
            assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)


def test_residual(net=None):
    logging.info('通道和特征图大小不变')
    blk = Residual(3, 3)
    X = torch.rand((4, 3, 6, 6))
    logging.info(blk(X).size())

    logging.info('通道和特征图大小改变')
    blk = Residual(3, 6, use_1x1conv=True, stride=2)
    logging.info(blk(X).shape)

    if net:
        logging.info('测试网络 ...')
        X = torch.rand((1, 1, 224, 224))
        for name, layer in net.named_children():
            X = layer(X)
            logging.info(f'{name} output shape:\t{X.size()}')


if __name__ == '__main__':
    batch_size, lr = 256, 0.001
    dataset = Dataset(batch_size)
    criterion = nn.CrossEntropyLoss()

    logging.info('BatchNormalization 从零开始实现 ...')
    lenet_bn(lr)

    logging.info('BatchNormalization 简洁实现 ...')
    lenet_bn(lr, simple=False)

    logging.info('ResNet ...')
    resnet = ResNet().net.to(device)
    optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
    train(resnet, dataset, criterion, optimizer)
