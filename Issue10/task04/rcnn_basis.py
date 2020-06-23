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


def rcnn(inputs, state, params):
    ''' 定义模型 '''
    


def test():
    logging.info('测试 one hot 编码...')
    x = torch.tensor([0, 2])
    x_one_hot = one_hot(x, vocab_size)
    logging.info(f'x one hot: {x_one_hot}')
    logging.info(f'shape of x one hot: {x_one_hot.shape}')
    logging.info(f'sum of x one hot: {x_one_hot.numpy().sum(axis=1)}')


if __name__ == '__main__':
    # 加载数据集
    corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()
    # 初始化参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

    # 测试
    test()
