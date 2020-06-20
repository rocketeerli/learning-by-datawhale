# -*- coding:utf-8 -*-
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def read_data(root='./data/jaychou', filename='jaychou_lyrics.txt'):
    ''' 读取数据集 '''
    with open(os.path.join(root, filename), 'r', encoding='utf-8') as f:
        corpus_chars = f.read()
    logging.info(len(corpus_chars))
    logging.info(corpus_chars[: 40])
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[: 10000]
    return corpus_chars


if __name__ == '__main__':
    corpus_chars = read_data()
