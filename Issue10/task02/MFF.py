# -*- coding: utf-8 -*-
import torch
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
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    features, labels = data_generator()
    logging.info(f'features: {features}')
    logging.info(f'labels: {labels}')
    show_data(features.numpy(), labels.numpy(), 'features', 'labels')
    MultivariateFunctionFitting()
