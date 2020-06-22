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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test()
