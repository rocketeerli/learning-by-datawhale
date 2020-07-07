# -*- coding:utf-8 -*-
import torch
import math
from torch import nn
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def SequenceMask(X, X_len, value=-1e6):
    ''' X 每行保留前多少个特征
    :param X: 替换前的数据
    :param X_len: 每行保留的数据数量
    :param value: 数据的替换值
    :return : 替换后的数据
    '''
    assert X.size(0) == X_len.size(0)
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float)[None, :] >= X_len[:, None]
    X[mask] = value
    return X


def masked_softmax(X, valid_length):
    # X: 3-D tensor, valid_length: 1-D or 2-D tensor
    softmax = nn.Softmax(dim=-1)
    if valid_length is None:
        return softmax(X)
    else:
        shape = X.shape
        if valid_length.dim() == 1:
            try:
                valid_length = torch.FloatTensor(valid_length.numpy().repeat(shape[1], axis=0))#[2,2,3,3]
            except:
                valid_length = torch.FloatTensor(valid_length.cpu().numpy().repeat(shape[1], axis=0))#[2,2,3,3]
        else:
            valid_length = valid_length.reshape((-1,))
        # fill masked elements with a large negative, whose exp is 0
        X = SequenceMask(X.reshape(-1, shape[-1]), valid_length)
        return softmax(X).reshape(shape)


class DotProductAttention(nn.Module): 
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # query: (batch_size, #queries, d)
    # key: (batch_size, #kv_pairs, d)
    # value: (batch_size, #kv_pairs, dim_v)
    # valid_length: either (batch_size, ) or (batch_size, xx)
    def forward(self, query, key, value, valid_length=None):
        d = query.shape[-1]
        # set transpose_b=True to swap the last two dimensions of key
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        print("attention_weight\n", attention_weights)
        return torch.bmm(attention_weights, value)


class MLPAttention(nn.Module):
    def __init__(self, units, ipt_dim, dropout, **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        # Use flatten=True to keep query's and key's 3-D shapes.
        self.W_k = nn.Linear(ipt_dim, units, bias=False)
        self.W_q = nn.Linear(ipt_dim, units, bias=False)
        self.v = nn.Linear(units, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_length):
        query, key = self.W_k(query), self.W_q(key)  # q 和 k 映射到相同的维度
        # print("size: ", query.size(), key.size())
        # expand query to (batch_size, #querys, 1, units), and key to
        # (batch_size, 1, #kv_pairs, units). Then plus them with broadcast.
        features = query.unsqueeze(2) + key.unsqueeze(1)  # 将 q 和 k 做加法
        # print("features:", features.size())  # (batch_size, #querys, #kv_pairs, units)
        scores = self.v(features).squeeze(-1)  # 转换成 1 维
        # print('v featrues: ', self.v(features))  # (batch_size, #querys, #kv_pairs, 1)
        # print('scores: ', scores)  # (batch_size, #querys, #kv_pairs)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))  # 做 softmax 屏蔽
        # print('attention_weights: ', attention_weights)
        return torch.bmm(attention_weights, value)


def test():
    logging.info('softmax 屏蔽 ...')
    mas_sof = masked_softmax(torch.rand((2, 2, 4), dtype=torch.float),
                             torch.FloatTensor([2, 3]))
    logging.info(f'mas_sof: {mas_sof}')

    logging.info('点积注意力 ...')
    atten = DotProductAttention(dropout=0)
    keys = torch.ones((2, 10, 2), dtype=torch.float)
    values = torch.arange(40, dtype=torch.float).view(1, 10, 4).repeat(2, 1, 1)
    dot_att = atten(torch.ones((2, 1, 2), dtype=torch.float), keys, values,
                    torch.FloatTensor([2, 6]))
    logging.info(f'dot attention output: {dot_att}')

    logging.info('多层感知机注意力 ...')
    atten = MLPAttention(ipt_dim=2, units=8, dropout=0)
    mlp_att = atten(torch.ones((2, 1, 2), dtype=torch.float), keys, values,
                    torch.FloatTensor([2, 6]))
    logging.info(f'mlp attention output: {mlp_att}')


if __name__ == '__main__':
    test()
