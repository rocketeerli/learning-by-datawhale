# -*- coding:utf-8 -*-
import collections
import re
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def read_data(root='./data/Books', filename='timemachine.txt'):
    with open(os.path.join(root, filename), 'r', encoding='utf-8') as f:
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines


def tokenize(sentences, token='word'):
    ''' 分词，把句子按单词或字符分开 '''
    if token == 'word':
        return [sentence.split(' ') for sentence in sentences]
    elif token == 'char':
        return [list(sentence) for sentence in sentences]
    logging.error(f'ERROR: unkown token type: {token}')


class Vocab():
    ''' 构建一个字典（vocabulary），将每个词映射到一个唯一的索引编号 '''
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = self.count_corpus(tokens)
        self.token_freqs = list(counter.item())

    @staticmethod
    def count_corpus(sentences):
        tokens = [tk for st in sentences for tk in st]
        return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数


if __name__ == '__main__':
    # 1. 读入文本
    lines = read_data()
    logging.info(f'lines number {len(lines)}')
    # 2. 分词
    tokens = tokenize(lines)
    logging.info(f'tokens: {tokens[:5]}')
    # 3. 建立字典
