# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import os


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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
