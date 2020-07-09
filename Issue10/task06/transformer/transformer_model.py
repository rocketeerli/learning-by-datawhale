# -*- coding:utf-8 -*-
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import DotProductAttention, plot, Encoder, Decoder, EncoderDecoder, load_data_nmt, try_gpu, train_s2s_ch9, predict_s2s_ch9
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


class MultiHeadAttention(nn.Module):
    ''' 多头注意力层 '''
    def __init__(self, input_size, hidden_size, num_heads, dropout, **kwargs):
        # hidden_size = hiden_size * num_heads ???
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(input_size, hidden_size, bias=False)
        self.W_k = nn.Linear(input_size, hidden_size, bias=False)
        self.W_v = nn.Linear(input_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, query, key, value, valid_length):
        # query, key, and value shape: (batch_size, seq_len, dim),
        # where seq_len is the length of input sequence
        # valid_length shape is either (batch_size, )
        # or (batch_size, seq_len).

        # Project and transpose query, key, and value from
        # (batch_size, seq_len, hidden_size * num_heads) to
        # (batch_size * num_heads, seq_len, hidden_size).

        query = transpose_qkv(self.W_q(query), self.num_heads)
        key = transpose_qkv(self.W_k(key), self.num_heads)
        value = transpose_qkv(self.W_v(value), self.num_heads)

        if valid_length is not None:
            # Copy valid_length by num_heads times
            device = valid_length.device
            valid_length = valid_length.cpu().numpy() if valid_length.is_cuda else valid_length.numpy()
            if valid_length.ndim == 1:
                valid_length = torch.FloatTensor(np.tile(valid_length, self.num_heads))
            else:
                valid_length = torch.FloatTensor(np.tile(valid_length, (self.num_heads, 1)))
            valid_length = valid_length.to(device)

        # print('query: ', query, '\nshape: ', query.shape)
        output = self.attention(query, key, value, valid_length)
        # print('output: ', output, '\nshape: ', output.shape)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    # Original X shape: (batch_size, seq_len, hidden_size * num_heads),
    # -1 means inferring its value, after first reshape, X shape:
    # (batch_size, seq_len, num_heads, hidden_size)
    # logging.info(f'X shape: {X.size()}')
    X = X.view(X.shape[0], X.shape[1], num_heads, -1)

    # After transpose, X shape: (batch_size, num_heads, seq_len, hidden_size)
    X = X.transpose(2, 1).contiguous()

    # Merge the first two dimensions. Use reverse=True to infer shape from
    # right to left.
    # output shape: (batch_size * num_heads, seq_len, hidden_size)
    output = X.view(-1, X.shape[2], X.shape[3])
    # logging.info(f'out shape: {X.size()}')
    return output


def transpose_output(X, num_heads):
    # A reversed version of transpose_qkv
    X = X.view(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(2, 1).contiguous()
    return X.view(X.shape[0], X.shape[1], -1)


class PositionWiseFFN(nn.Module):
    ''' 基于位置的前馈神经网络 '''
    def __init__(self, input_size, ffn_hidden_size, hidden_size_out, **kwargs):
        super().__init__(**kwargs)
        self.ffn_1 = nn.Linear(input_size, ffn_hidden_size)
        self.ffn_2 = nn.Linear(ffn_hidden_size, hidden_size_out)

    def forward(self, X):
        return self.ffn_2(F.relu(self.ffn_1(X)))


class AddNorm(nn.Module):
    ''' 相加归一层 '''
    def __init__(self, hidden_size, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, X, Y):
        return self.norm(self.dropout(Y) + X)


class PositionalEncoding(nn.Module):
    ''' 位置编码 '''
    def __init__(self, embedding_size, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = np.zeros((1, max_len, embedding_size))
        X = np.arange(0, max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, embedding_size, 2)/embedding_size)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        self.P = torch.FloatTensor(self.P)

    def forward(self, X):
        if X.is_cuda and not self.P.is_cuda:
            self.P = self.P.cuda()
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)


class EncoderBlock(nn.Module):
    ''' 编码器
    包含一个多头注意力层，一个position-wise FFN，和两个 Add and Norm 层
    '''
    def __init__(self, embedding_size, ffn_hidden_size, num_heads,
                 dropout, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(embedding_size, embedding_size, num_heads, dropout)
        self.addnorm_1 = AddNorm(embedding_size, dropout)
        self.ffn = PositionWiseFFN(embedding_size, ffn_hidden_size, embedding_size)
        self.addnorm_2 = AddNorm(embedding_size, dropout)

    def forward(self, X, valid_length):
        Y = self.addnorm_1(X, self.attention(X, X, X, valid_length))
        return self.addnorm_2(Y, self.ffn(Y))


class TransformerEncoder(Encoder):
    def __init__(self, vocab_size, embedding_size, ffn_hidden_size,
                 num_heads, num_layers, dropout, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoding = PositionalEncoding(embedding_size, dropout)
        self.blks = nn.ModuleList()
        for i in range(num_layers):
            self.blks.append(
                EncoderBlock(embedding_size, ffn_hidden_size,
                             num_heads, dropout))

    def forward(self, X, valid_length, *args):
        # 把嵌入向量乘以 根号d 以防止其值过小。
        X = self.pos_encoding(self.embed(X) * math.sqrt(self.embedding_size))
        for blk in self.blks:
            X = blk(X, valid_length)
        return X


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, ffn_hidden_size, num_heads, dropout, i, **kwargs):
        super().__init__(**kwargs)
        self.i = i
        self.attention_1 = MultiHeadAttention(embedding_size, embedding_size, num_heads, dropout)
        self.addnorm_1 = AddNorm(embedding_size, dropout)
        self.attention_2 = MultiHeadAttention(embedding_size, embedding_size, num_heads, dropout)
        self.addnorm_2 = AddNorm(embedding_size, dropout)
        self.ffn = PositionWiseFFN(embedding_size, ffn_hidden_size, embedding_size)
        self.addnorm_3 = AddNorm(embedding_size, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_length = state[0], state[1]

        # state[2][self.i] stores all the previous t-1 query state of layer-i
        # len(state[2]) = num_layers
        # If training:
        #     state[2] is useless.
        # If predicting:
        #     In the t-th timestep:
        #         state[2][self.i].shape = (batch_size, t-1, hidden_size)
        # Demo:
        # love dogs ! [EOS]
        #  |    |   |   |
        #   Transformer 
        #    Decoder
        #  |   |   |   |
        #  I love dogs !

        if state[2][self.i] is None:
            key_values = X
        else:
            # shape of key_values = (batch_size, t, hidden_size)
            key_values = torch.cat((state[2][self.i], X), dim=1) 
        state[2][self.i] = key_values

        if self.training:
            batch_size, seq_len, _ = X.shape
            # Shape: (batch_size, seq_len), the values in the j-th column are j+1
            valid_length = torch.FloatTensor(np.tile(np.arange(1, seq_len+1), (batch_size, 1))) 
            valid_length = valid_length.to(X.device)
        else:
            valid_length = None

        X2 = self.attention_1(X, key_values, key_values, valid_length)
        Y = self.addnorm_1(X, X2)
        Y2 = self.attention_2(Y, enc_outputs, enc_outputs, enc_valid_length)
        Z = self.addnorm_2(Y, Y2)
        return self.addnorm_3(Z, self.ffn(Z)), state


class TransformerDecoder(Decoder):
    def __init__(self, vocab_size, embedding_size, ffn_hidden_size,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoding = PositionalEncoding(embedding_size, dropout)
        self.blks = nn.ModuleList()
        for i in range(num_layers):
            self.blks.append(
                DecoderBlock(embedding_size, ffn_hidden_size, num_heads,
                             dropout, i))
        self.dense = nn.Linear(embedding_size, vocab_size)

    def init_state(self, enc_outputs, enc_valid_length, *args):
        return [enc_outputs, enc_valid_length, [None]*self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embed(X) * math.sqrt(self.embedding_size))
        for blk in self.blks:
            X, state = blk(X, state)
        return self.dense(X), state


def test():
    logging.info('多头注意力（Multi-head Attention Layers）...')
    cell = MultiHeadAttention(5, 9, 3, 0.5)
    X = torch.ones((2, 4, 5))
    valid_length = torch.FloatTensor([2, 3])
    logging.info(f'multi head attention outpits shape : {cell(X, X, X, valid_length).shape}')

    logging.info('前馈神经网络 FFN ...')
    ffn = PositionWiseFFN(4, 4, 8)
    out = ffn(torch.ones((2, 3, 4)))
    logging.info(f'out: {out} \n shape of out: {out.shape}')

    logging.info('LayerNorm 与 BatchNorm 进行对比 ...')
    layernorm = nn.LayerNorm(normalized_shape=2, elementwise_affine=True)
    batchnorm = nn.BatchNorm1d(num_features=2, affine=True)
    X = torch.FloatTensor([[1, 2], [3, 4]])
    print('layer norm:', layernorm(X))
    print('batch norm:', batchnorm(X))

    logging.info('相加归一层 AddNorm ...')
    add_norm = AddNorm(4, 0.5)
    output_addnorm = add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape
    logging.info(f'output of addnorm: {output_addnorm}')

    logging.info('位置编码 ...')
    pe = PositionalEncoding(20, 0)
    Y = pe(torch.zeros((1, 100, 20))).numpy()
    # plot(np.arange(100), Y[0, :, 4:8].T, figsize=(6, 2.5),
    #          legend=["dim %d" % p for p in [4, 5, 6, 7]])

    logging.info('编码器模块 EncoderBlock ...')
    X = torch.ones((2, 100, 24))
    encoder_blk = EncoderBlock(24, 48, 8, 0.5)
    logging.info(f'endcoer output shape: {encoder_blk(X, valid_length).shape}')

    logging.info('Transformer 编码器 ...')
    encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
    logging.info(f'encoder output shape: {encoder(torch.ones((2, 100)).long(), valid_length).shape}')


if __name__ == '__main__':
    # test()

    embed_size, embedding_size, num_layers, dropout = 32, 32, 2, 0.05
    batch_size, num_steps = 64, 10
    lr, num_epochs, ctx = 0.005, 250, try_gpu()
    logging.info(f'using {ctx} ...')
    num_hiddens, num_heads = 64, 4

    src_vocab, tgt_vocab, train_iter = load_data_nmt(batch_size, num_steps)

    encoder = TransformerEncoder(
        len(src_vocab), embedding_size, num_hiddens, num_heads, num_layers,
        dropout)
    decoder = TransformerDecoder(
        len(src_vocab), embedding_size, num_hiddens, num_heads, num_layers,
        dropout)
    model = EncoderDecoder(encoder, decoder)
    train_s2s_ch9(model, train_iter, lr, num_epochs, ctx)

    # 测试模型
    model.eval()
    for sentence in ['Go .', 'Wow !', "I'm OK .", 'I won !']:
        print(sentence + ' => ' + predict_s2s_ch9(
            model, sentence, src_vocab, tgt_vocab, num_steps, ctx))
