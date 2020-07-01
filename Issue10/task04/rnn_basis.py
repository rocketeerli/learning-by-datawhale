# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import time
import math
from language_model import load_data_jay_lyrics
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def one_hot(x, n_class, dtype=torch.float32):
    '''one hot 编码
    :param x: 一个时序序列（索引方式）
    :param n_class: 索引类别的数目
    '''
    result = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    result.scatter_(1, x.long().view(-1, 1), 1)  # result[i, x[i, 0]] = 1
    return result


def to_onehot(X, n_class):
    '''
    :param X: (batch_size, timestep)
    :param n_class: one hot 编码维度
    :return: (timestep, batch_size, n_class)
    '''
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


def get_params():
    ''' 初始化模型参数 '''
    def _one(shape):
        param = torch.zeros(shape, device=device, dtype=torch.float32)
        nn.init.normal_(param, 0, 0.01)
        return nn.Parameter(param)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nn.Parameter(torch.zeros(num_hiddens, device=device))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nn.Parameter(torch.zeros(num_outputs, device=device))
    return (W_xh, W_hh, b_h, W_hq, b_q)


def rnn(inputs, state, params):
    ''' 定义模型 '''
    # inputs 和 outputs 皆为 num_steps 个形状为 (batch_size, vocab_size) 的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


def init_rnn_state(batch_size, num_hiddens, device):
    ''' 初始化 rnn 的隐藏变量，即H0 '''
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def grad_clipping(params, theta, device):
    ''' 梯度裁剪 '''
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def test():
    logging.info('测试 one hot 编码...')
    x = torch.tensor([0, 2])
    x_one_hot = one_hot(x, vocab_size)
    logging.info(f'x one hot: {x_one_hot}')
    logging.info(f'shape of x one hot: {x_one_hot.shape}')
    logging.info(f'sum of x one hot: {x_one_hot.numpy().sum(axis=1)}')

    logging.info('测试 one hot 编码一个 batch 的数据...')
    X = torch.arange(10).view(2, 5)
    inputs = to_onehot(X, vocab_size)
    logging.info(f'时步 length of inputs: {len(inputs)}')
    logging.info(f'每个时步的输入 shape of inputs[0]: {inputs[0].shape}')

    logging.info('观察输出结果的个数，以及输出层和隐藏层的形状...')
    logging.info(f'隐藏层维度: {num_hiddens} \t 输入和输出的维度: {vocab_size}')
    state_init = init_rnn_state(X.shape[0], num_hiddens, device)
    inputs = to_onehot(X.to(device), vocab_size)
    params = get_params()
    outputs, state_new = rnn(inputs, state_init, params)
    logging.info(f'inputs length: {len(inputs)} \t shape: {inputs[0].shape}')
    logging.info(f'outputs length: {len(outputs)} \t shape: {outputs[0].shape}')
    logging.info(f'state init length: {len(state_init)} \t shape: {state_init[0].shape}')
    logging.info(f'state new length: {len(state_new)} \t shape: {state_new[0].shape}')


if __name__ == '__main__':
    # 加载数据集
    corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()
    # 初始化参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

    # 测试
    test()
