# -*- coding:utf-8 -*-
import collections
import re
import os
import logging
import spacy
from nltk.tokenize import word_tokenize
from nltk import data

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
        self.token_freqs = counter.items()
        self.index_to_token = []
        if use_special_tokens:
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.index_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.index_to_token += ['']
        self.index_to_token += [token for token, freq in self.token_freqs
                            if freq >= min_freq and token not in self.index_to_token]
        self.token_to_indx = dict()
        for index, token in enumerate(self.index_to_token):
            self.token_to_indx[token] = index

    def __len__(self):
        return len(self.index_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_indx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.index_to_token[indices]
        return [self.index_to_token[index] for index in indices]

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
    vocab = Vocab(tokens)
    logging.info(list(vocab.token_to_indx.items())[0:10])
    for i in range(8, 10):
        logging.info(f'words: {tokens[i]}')
        logging.info(f'indices: {vocab[tokens[i]]}')
    # 高级的分词方法
    # spaCy
    text = "Mr. Chen doesn't agree with my suggestion. Mr. Chen does agree with my suggestion."
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    logging.info(f'type of doc {type(doc)}')
    logging.info(f'spacy: {[token.text for token in doc]}')
    logging.info(list(doc.sents))
    # NTLK
    data.path.append('./data/nltk_data')
    logging.info(f'ntlk: {word_tokenize(text)}')
