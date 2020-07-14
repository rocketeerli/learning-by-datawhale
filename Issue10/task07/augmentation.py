# -*- coding:utf-8 -*-
import torch
from torch.utils.data import DataLoader
import torchvision
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from utils import train, resnet18
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)


def load_cifar10(is_train, augs, batch_size, root=None):
    root = CIFAR_ROOT_PATH if root is None else root
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, resnet18(10)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)


def test():
    root = './data'
    img = Image.open(os.path.join(root, 'img/cat1.jpg'))
    plt.imshow(img)
    plt.show()

    logging.info('一半概率的图像水平（左右）和垂直（上下）翻转 ...')
    apply(img, torchvision.transforms.RandomHorizontalFlip())
    plt.show()
    apply(img, torchvision.transforms.RandomVerticalFlip())
    plt.show()

    logging.info('随机裁剪')
    shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
    apply(img, shape_aug)
    plt.show()

    logging.info('变化颜色——亮度')
    apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))
    plt.show()
    logging.info('变化颜色——色调')
    apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))
    plt.show()
    logging.info('变化颜色——对比度')
    apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0))
    plt.show()
    logging.info('同时随机变化图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）')
    color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    apply(img, color_aug)
    plt.show()

    logging.info('叠加多个图像增广方法')
    augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
    apply(img, augs)
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0 if sys.platform.startswith('win32') else 4
    logging.info(f'use {device} ...')

    # test()

    CIFAR_ROOT_PATH = './data/cifar'
    all_imges = torchvision.datasets.CIFAR10(train=True, root=CIFAR_ROOT_PATH, download=True)
    show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=0.8)
    plt.show()

    flip_aug = torchvision.transforms.Compose([
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.ToTensor()])

    no_aug = torchvision.transforms.Compose([
         torchvision.transforms.ToTensor()])

    train_with_data_aug(flip_aug, no_aug)
