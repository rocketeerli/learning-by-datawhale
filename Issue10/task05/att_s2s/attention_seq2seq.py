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


if __name__ == '__main__':
    test()
