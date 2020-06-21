# -*- coding:utf-8 -*-
import torch
import random
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def load_data_jay_lyrics(root='./data/jaychou', filename='jaychou_lyrics.txt'):
    # 读取数据
    with open(os.path.join(root, filename), 'r', encoding='utf-8') as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    # 建立数据集字符索引
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    ''' 随机采样
    :param corpus_indices: 序列数据集的索引
    :param batch_size: 每个小批量的样本数
    :param num_steps: 每个样本所包含的时间步数
    '''
    # 减 1 是因为对于长度为n的序列，X最多只有包含其中的前n - 1个字符
    num_examples = (len(corpus_indices) - 1) // num_steps  # 下取整，得到不重叠情况下的样本个数
    example_indices = [i * num_steps for i in range(num_examples)]  # 每个样本的第一个字符在corpus_indices中的下标
    random.shuffle(example_indices)

    def _data(i):
        # 返回从i开始的长为num_steps的序列
        return corpus_indices[i: i + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(0, num_examples, batch_size):
        # 每次选出batch_size个随机样本
        batch_indices = example_indices[i: i + batch_size]  # 当前batch的各个样本的首字符的下标
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield torch.tensor(X, device=device), torch.tensor(Y, device=device)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    ''' 相邻采样 '''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_len = len(corpus_indices) // batch_size * batch_size  # 保留下来的序列的长度
    corpus_indices = corpus_indices[: corpus_len]  # 仅保留前corpus_len个字符
    indices = torch.tensor(corpus_indices, device=device)
    indices = indices.view(batch_size, -1)  # resize成(batch_size, )
    batch_num = (indices.shape[1] - 1) // num_steps
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


def test():
    my_seq = list(range(30))
    logging.info('随机采样...')
    for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
        logging.info(f'\nX: {X.data}\nY:{Y.data}')
    logging.info('相邻采样...')
    for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
        logging.info(f'\nX: {X.data}\nY:{Y.data}')


if __name__ == '__main__':
    logging.info('加载数据集，并建立数据集字符索引...')
    corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()

    logging.info('测试采样方式...')
    test()
