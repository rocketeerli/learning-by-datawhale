# -*- coding:utf-8 -*-
import collections
import zipfile
from base import Vocab
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data
import logging
from utils import set_figsize, plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def preprocess_raw(text):
    ''' 整理数据集，替换掉无用字符 '''
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    out = ''
    for i, char in enumerate(text.lower()):
        if char in (',', '!', '.') and i > 0 and text[i-1] != ' ':
            out += ' '
        out += char
    return out


def build_vocab(tokens):
    ''' 创建单词数据集 '''
    tokens = [token for line in tokens for token in line]
    return Vocab(tokens, min_freq=3, use_special_tokens=True)


def pad(line, max_len, padding_token):
    ''' 统一句子的长度，截断 or 填充 '''
    if len(line) > max_len:
        return line[:max_len]
    return line + [padding_token] * (max_len - len(line))


def build_array(lines, vocab, max_len, is_source):
    ''' 将字符形式的单词转换成向量形式
    :param lines: 每个句子的字符形式
    :param vocab: 数据的单词集
    :param max_len: 每个单词的最大长度
    :param is_source: 在句子前后添加开始和结束符
    :return array: 每个句子的向量形式
    :return valid_len: 每个句子的有效长度
    '''
    lines = [vocab[line] for line in lines]
    if not is_source:
        lines = [[vocab.bos] + line + [vocab.eos] for line in lines]
    array = torch.tensor([pad(line, max_len, vocab.pad) for line in lines])
    valid_len = (array != vocab.pad).sum(1)  # 第一个维度
    return array, valid_len


def load_data_nmt(batch_size, max_len):
    src_vocab, tgt_vocab = build_vocab(source), build_vocab(target)
    src_array, src_valid_len = build_array(source, src_vocab, max_len, True)
    tgt_array, tgt_valid_len = build_array(target, tgt_vocab, max_len, False)
    train_data = data.TensorDataset(src_array, src_valid_len, tgt_array, tgt_valid_len)
    train_iter = data.DataLoader(train_data, batch_size, shuffle=True)
    return src_vocab, tgt_vocab, train_iter


if __name__ == '__main__':
    logging.info('读取数据文件...')
    with open('./data/fraeng3479/fra.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    logging.info(raw_text[0:1000])
    text = preprocess_raw(raw_text)
    logging.info(text[0:1000])

    logging.info('分词...')
    num_examples = 50000
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) >= 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    logging.info(f'source: {source[:3]}')
    logging.info(f'target: {target[:3]}')

    # set_figsize()
    # plt.hist([[len(l) for l in source], [len(l) for l in target]], label=['source', 'target'])
    # plt.legend(loc='upper right')
    # plt.show()

    src_vocab = build_vocab(source)
    logging.info(f'number of tokens: {len(src_vocab)}')

    logging.info(f'after padding: {pad(src_vocab[source[0]], 10, src_vocab.pad)}')

    src_vocab, tgt_vocab, train_iter = load_data_nmt(batch_size=2, max_len=8)
    for X, X_valid_len, Y, Y_valid_len, in train_iter:
        print('X =', X.type(torch.int32), '\nValid lengths for X =', X_valid_len,
                     '\nY =', Y.type(torch.int32), '\nValid lengths for Y =', Y_valid_len)
        break
