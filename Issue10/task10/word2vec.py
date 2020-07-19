# -*- coding:utf-8 -*-
import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data


def read_data():
    with open('./data/ptb_train/ptb.train.txt', 'r') as f:
        lines = f.readlines()  # 该数据集中句子以换行符为分割
        raw_dataset = [st.split() for st in lines]  # st是sentence的缩写，单词以空格为分割
    print('# sentences: %d' % len(raw_dataset))
    # 对于数据集的前3个句子，打印每个句子的词数和前5个词
    # 句尾符为 '' ，生僻词全用 '' 表示，数字则被替换成了 'N'
    for st in raw_dataset[:3]:
        print('# tokens:', len(st), st[:5])

    counter = collections.Counter([tk for st in raw_dataset for tk in st])  # tk是token的缩写
    # items()  # 转化成(元素，计数值)组成的列表
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))  # 只保留在数据集中至少出现5次的词
    idx_to_token = [tk for tk, _ in counter.items()]
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
    dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
               for st in raw_dataset]  # raw_dataset中的单词在这一步被转换为对应的idx
    num_tokens = sum([len(st) for st in dataset])
    print('# tokens: %d' % num_tokens)
    return counter, idx_to_token, token_to_idx, dataset, num_tokens


def discard(idx):
    '''
    @params:
        idx: 单词的下标
    @return: True/False 表示是否丢弃该单词
    '''
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)


def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))


def get_centers_and_contexts(dataset, max_window_size):
    '''
    @params:
        dataset: 数据集为句子的集合，每个句子则为单词的集合，此时单词已经被转换为相应数字下标
        max_window_size: 背景词的词窗大小的最大值
    @return:
        centers: 中心词的集合
        contexts: 背景词窗的集合，与中心词对应，每个背景词窗则为背景词的集合
    '''
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size) # 随机选取背景词窗大小
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


if __name__ == '__main__':
    counter, idx_to_token, token_to_idx, dataset, num_tokens = read_data()

    # 二次采样
    subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
    print('# tokens: %d' % sum([len(st) for st in subsampled_dataset]))
    print(compare_counts('the'))
    print(compare_counts('join'))

    # 提取中心词和背景词
    all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)
    tiny_dataset = [list(range(7)), list(range(7, 10))]
    print('dataset', tiny_dataset)
    for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
        print('center', center, 'has contexts', context)
