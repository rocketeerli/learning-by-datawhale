# -*- coding:utf-8 -*-
import torch
from torch import nn
import logging
from utils import load_data_jay_lyrics, RNNModel
from utils import train_and_predict_rnn, train_and_predict_rnn_pytorch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def train():
    num_hiddens=256
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e-2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

    logging.info('深度循环神经网络...')
    logging.info('2 层深度循环神经网络...')
    gru_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens, num_layers=2)
    model = RNNModel(gru_layer, vocab_size).to(device)
    train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes)
    logging.info('6 层深度循环神经网络...')
    gru_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens, num_layers=6)
    model = RNNModel(gru_layer, vocab_size).to(device)
    train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes)

    logging.info('双向循环神经网络...')

    gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens, bidirectional=True)
    model = RNNModel(gru_layer, vocab_size).to(device)
    train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes)


if __name__ == '__main__':
    logging.info('加载数据集并初始化全局变量...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    logging.info(f'will use: {device}')

    train()
