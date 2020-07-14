# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import time
import math
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results


def show_trace_2d(f, results):
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')


def get_data_ch7():
    data = np.genfromtxt('./data/airfoil4755/airfoil_self_noise.dat', delimiter='\t')
    # logging.info(f'data type: {type(data)} \t shape: {data.shape}')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
        torch.tensor(data[:1500, -1], dtype=torch.float32)  # 前1500个样本(每个样本5个特征)


def linreg(X, w, b):
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    # 注意这里返回的是向量, 另外, pytorch 里的 MSELoss 并没有除以 2
    return ((y_hat - y.view(y_hat.size())) ** 2) / 2


def train_ch7(optimizer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = linreg, squared_loss
    w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32),
                           requires_grad=True)
    b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()
    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(TensorDataset(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()  # 使用平均损失
            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            optimizer_fn([w, b], states, hyperparams)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # 每100个样本记录下当前训练误差
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')


def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams, features, labels,
                      batch_size=10, num_epochs=2):
    # 初始化模型
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1)
    )
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2
    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(TensorDataset(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 除以2是为了和train_ch7保持一致, 因为squared_loss中除了2
            l = loss(net(X).view(-1), y) / 2 
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def test_momentum():
    logging.info('test momentum ...')
    eta = 0.6

    def gd_2d(x1, x2, s1, s2):
        return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

    # show_trace_2d(f_2d, train_2d(gd_2d))
    # plt.show()

    logging.info('使用 momentum 的方法 ...')

    def momentum_2d(x1, x2, v1, v2):
        v1 = beta * v1 + eta * 0.2 * x1
        v2 = beta * v2 + eta * 4 * x2
        return x1 - v1, x2 - v2, v1, v2

    eta, beta = 0.6, 0.5
    # show_trace_2d(f_2d, train_2d(momentum_2d))
    # plt.show()

    logging.info('动量方法做训练 手动实现 ...')

    def init_momentum_states():
        v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
        v_b = torch.zeros(1, dtype=torch.float32)
        return (v_w, v_b)

    def sgd_momentum(params, states, hyperparams):
        for p, v in zip(params, states):
            v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
            p.data -= v.data

    train_ch7(sgd_momentum, init_momentum_states(),
              {'lr': 0.004, 'momentum': 0.9}, features, labels)
    plt.show()

    logging.info('动量方法做训练 PyTorch 实现 ...')
    train_pytorch_ch7(torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9},
                      features, labels)
    plt.show()


def test_AdaGrad():
    logging.info('体验 AdaGrad ...')
    eta = 2

    def adagrad_2d(x1, x2, s1, s2):
        g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # 前两项为自变量梯度
        s1 += g1 ** 2
        s2 += g2 ** 2
        x1 -= eta / math.sqrt(s1 + eps) * g1
        x2 -= eta / math.sqrt(s2 + eps) * g2
        return x1, x2, s1, s2

    show_trace_2d(f_2d, train_2d(adagrad_2d))
    plt.show()

    logging.info('AdaGrad Implement ...')
    features, labels = get_data_ch7()

    def init_adagrad_states():
        s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
        s_b = torch.zeros(1, dtype=torch.float32)
        return (s_w, s_b)

    def adagrad(params, states, hyperparams):
        eps = 1e-6
        for p, s in zip(params, states):
            s.data += (p.grad.data**2)
            p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)

    train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)
    plt.show()

    logging.info('AdaGrad PyTorch ...')
    train_pytorch_ch7(torch.optim.Adagrad, {'lr': 0.1}, features, labels)
    plt.show()


def test_RMSProp():
    logging.info('体验 RMSProp ...')
    alpha, beta = 0.4, 0.9

    def rmsprop_2d(x1, x2, s1, s2):
        g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
        s1 = beta * s1 + (1 - beta) * g1 ** 2
        s2 = beta * s2 + (1 - beta) * g2 ** 2
        x1 -= alpha / math.sqrt(s1 + eps) * g1
        x2 -= alpha / math.sqrt(s2 + eps) * g2
        return x1, x2, s1, s2

    show_trace_2d(f_2d, train_2d(rmsprop_2d))
    plt.show()

    logging.info('RMSProp Implement ...')

    def init_rmsprop_states():
        s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
        s_b = torch.zeros(1, dtype=torch.float32)
        return (s_w, s_b)

    def rmsprop(params, states, hyperparams):
        gamma, eps = hyperparams['beta'], 1e-6
        for p, s in zip(params, states):
            s.data = gamma * s.data + (1 - gamma) * (p.grad.data)**2
            p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)
    train_ch7(rmsprop, init_rmsprop_states(), {'lr': 0.01, 'beta': 0.9},
              features, labels)
    plt.show()

    logging.info('RMSProp wih PyTorch ...')
    train_pytorch_ch7(torch.optim.RMSprop, {'lr': 0.01, 'alpha': 0.9},
                      features, labels)
    plt.show()


def test_AdaDelta():
    logging.info('AdaDelta Implement ...')

    def init_adadelta_states():
        s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
        delta_w, delta_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
        return ((s_w, delta_w), (s_b, delta_b))

    def adadelta(params, states, hyperparams):
        rho, eps = hyperparams['rho'], 1e-5
        for p, (s, delta) in zip(params, states):
            s[:] = rho * s + (1 - rho) * (p.grad.data**2)
            g = p.grad.data * torch.sqrt((delta + eps) / (s + eps))
            p.data -= g
            delta[:] = rho * delta + (1 - rho) * g * g
    train_ch7(adadelta, init_adadelta_states(), {'rho': 0.9}, features, labels)
    plt.show()

    logging.info('AdaDelta with PyTorch ...')
    train_pytorch_ch7(torch.optim.Adadelta, {'rho': 0.9}, features, labels)
    plt.show()


def test_Adam():
    logging.info('Adam Implement ...')

    def init_adam_states():
        v_w, v_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
        s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
        return ((v_w, s_w), (v_b, s_b))

    def adam(params, states, hyperparams):
        beta1, beta2, eps = 0.9, 0.999, 1e-6
        for p, (v, s) in zip(params, states):
            v[:] = beta1 * v + (1 - beta1) * p.grad.data
            s[:] = beta2 * s + (1 - beta2) * p.grad.data**2
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p.data -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
        hyperparams['t'] += 1
    train_ch7(adam, init_adam_states(), {'lr': 0.01, 't': 1}, features, labels)
    plt.show()

    logging.info('Adam with PyTorch ...')
    train_pytorch_ch7(torch.optim.Adam, {'lr': 0.01}, features, labels)
    plt.show()


def test():
    logging.info('带动量的小批量梯度下降 ...')
    test_momentum()

    logging.info('AdaGrad ...')
    test_AdaGrad()

    logging.info('RMSProp ...')
    test_RMSProp()

    logging.info('AdaDelta ...')
    test_AdaDelta()

    logging.info('Adam ...')
    test_Adam()


if __name__ == '__main__':
    features, labels = get_data_ch7()
    test()
