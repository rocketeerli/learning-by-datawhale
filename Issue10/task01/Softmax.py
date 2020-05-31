# -*- coding:utf-8  -*-
import torch
import torchvision
import torchvision.transforms as transforms
import time
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s')

mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


if __name__ == '__main__':
    logging.info(type(mnist_train))
    logging.info(f'mnist train length: {len(mnist_train)} \t mnist test length: {len(mnist_test)}')
