# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch import nn
import logging
from utils import load_data_jay_lyrics, RNNModel, train_and_predict_rnn, train_and_predict_rnn_pytorch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def get_params():
    ''' 初始化 GRU 网络的参数 '''
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32) #正态分布
        return torch.nn.Parameter(ts, requires_grad=True)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))
    W_xz, W_hz, b_z = _three()  # 更新门参数
    W_xr, W_hr, b_r = _three()  # 重置门参数
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])


def init_gru_state(batch_size, num_hiddens, device):
    ''' 隐藏状态初始化 '''
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def gru(inputs, state, params):
    assert len(params) == 11
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + R * torch.matmul(H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


def train_gru():
    logging.info('GRU original ...')
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)

    logging.info('GRU with PyTorch ...')
    lr = 1e-2  # 注意调整学习率
    gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
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

    logging.info('GRU ...')
    train_gru()
