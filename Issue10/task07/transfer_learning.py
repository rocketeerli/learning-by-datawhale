# -*- coding:utf-8 -*-
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
from utils import show_images, train
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


def test_readdata():
    logging.info('读取数据集 ...')
    train_imgs = ImageFolder(os.path.join(data_dir, 'train'))
    test_imgs = ImageFolder(os.path.join(data_dir, 'test'))

    # 查看图像
    hotdogs = [train_imgs[i][0] for i in range(8)]
    not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
    show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
    plt.show()


def test_compare():
    ''' 作为对比，我们定义一个相同的模型，但将它的所有模型参数都初始化为随机值。由于整个模型都需要从头训练，我们可以使用较大的学习率。 '''
    scratch_net = models.resnet18(pretrained=False, num_classes=2)
    lr = 0.1
    optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
    train_fine_tuning(scratch_net, optimizer)


if __name__ == '__main__':
    root = './data'
    data_dir = os.path.join(root, 'hotdog')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test_readdata()

    # 数据预处理
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_augs = transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    test_augs = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            normalize
        ])

    # 加载预训练模型
    pretrained_net = models.resnet18(pretrained=False)
    pretrained_net.load_state_dict(torch.load(os.path.join(root, 'resnet-18/resnet18-5c106cde.pth')))
    logging.info(pretrained_net.fc)

    # 随机初始化最后一层参数
    pretrained_net.fc = nn.Linear(512, 2)
    logging.info(pretrained_net.fc)

    output_params = list(map(id, pretrained_net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
    lr = 0.01
    # fc中的随机初始化参数一般需要更大的学习率从头训练
    optimizer = optim.SGD([{'params': feature_params},
                           {'params': pretrained_net.fc.parameters(), 'lr': lr*10}],
                           lr=lr, weight_decay=0.001)

    # 训练模型
    logging.info('随机初始化最后一层 fc ...')
    train_fine_tuning(pretrained_net, optimizer)

    logging.info('所有模型参数都初始化为随机值 ...')
    test_compare()
