# -*- coding:utf-8 -*-
import torch
import math
from torch import nn, optim
import logging
from utils import load_data_nmt, try_gpu, grad_clipping_nn
from sequence_to_sequence import Encoder, Decoder, EncoderDecoder, Seq2SeqEncoder
import time

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
    mask = torch.arange(maxlen, dtype=torch.float)[None, :].to(X_len.device) >= X_len.float()[:, None]
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


class Seq2SeqAttentionDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention_cell = MLPAttention(num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_len, *args):
        outputs, hidden_state = enc_outputs
        # print("first:",outputs.size(),hidden_state[0].size(),hidden_state[1].size())
        # Transpose outputs to (batch_size, seq_len, hidden_size)
        return (outputs.permute(1, 0, -1), hidden_state, enc_valid_len)
        # outputs.swapaxes(0, 1)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_len = state
        # print("X.size", X.size())
        X = self.embedding(X).transpose(0, 1)
        # print("Xembeding.size2", X.size())
        outputs = []
        for l, x in enumerate(X):  # 遍历每一个时步
            # print(f"\n{l}-th token")
            # print("x.first.size(): ", x.size())
            # query shape: (batch_size, 1, hidden_size)
            # select hidden state of the last rnn layer as query
            # print('hidden_state length: ', len(hidden_state), 'hidden_state shape: ', hidden_state[0].shape)
            query = hidden_state[0][-1].unsqueeze(1)  # np.expand_dims(hidden_state[0][-1], axis=1)
            # context has same shape as query
            # print("query enc_outputs, enc_outputs:\n", query.size(), enc_outputs.size())
            context = self.attention_cell(query, enc_outputs, enc_outputs, enc_valid_len)
            # Concatenate on the feature dimension
            # print("context.size:", context.size())
            x = torch.cat((context, x.unsqueeze(1)), dim=-1)
            # Reshape x to (1, batch_size, embed_size+hidden_size)
            # print("rnn: ", x.size(), len(hidden_state))
            out, hidden_state = self.rnn(x.transpose(0, 1), hidden_state)
            outputs.append(out)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.transpose(0, 1), [enc_outputs, hidden_state,
                                         enc_valid_len]


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_length):
        # the sample weights shape should be (batch_size, seq_len)
        # logging.info(f'pred : {pred.size()} \t label: {label.size()} \t valid_length: {valid_length.size()}')
        # print('pred: ', pred, '\n', 'label: ', label, 'valid_length: ', valid_length)
        weights = torch.ones_like(label)
        weights = SequenceMask(weights, valid_length, 0).float()
        self.reduction = 'none'
        output = super().forward(pred.transpose(1, 2), label)
        return (output*weights).mean(dim=1)


def train_s2s_ch9(model, data_iter, lr, num_epochs, device):  # Saved in d2l
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    tic = time.time()
    for epoch in range(1, num_epochs+1):
        l_sum, num_tokens_sum = 0.0, 0.0
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_vlen, Y, Y_vlen = [x.to(device) for x in batch]
            Y_input, Y_label, Y_vlen = Y[:, :-1], Y[:, 1:], Y_vlen-1
            
            Y_hat, _ = model(X, Y_input, X_vlen, Y_vlen)
            l = loss(Y_hat, Y_label, Y_vlen).sum()
            l.backward()

            with torch.no_grad():
                grad_clipping_nn(model, 5, device)
            num_tokens = Y_vlen.sum().item()
            optimizer.step()
            l_sum += l.sum().item()
            num_tokens_sum += num_tokens
        if epoch % 50 == 0:
            print("epoch {0:4d}, loss {1:.3f}, time {2:.1f} sec".format( 
                  epoch, (l_sum/num_tokens_sum), time.time()-tic))
            tic = time.time()


def predict_s2s_ch9(model, src_sentence, src_vocab, tgt_vocab, max_len, device):
    src_tokens = src_vocab[src_sentence.lower().split(' ')]
    src_len = len(src_tokens)
    if src_len < max_len:
        src_tokens += [src_vocab.pad] * (max_len - src_len)
    enc_X = torch.tensor(src_tokens, device=device)
    enc_valid_length = torch.tensor([src_len], device=device)
    # use expand_dim to add the batch_size dimension.
    enc_outputs = model.encoder(enc_X.unsqueeze(dim=0), enc_valid_length)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_length)
    dec_X = torch.tensor([tgt_vocab.bos], device=device).unsqueeze(dim=0)
    predict_tokens = []
    for _ in range(max_len):
        Y, dec_state = model.decoder(dec_X, dec_state)
        # The token with highest score is used as the next time step input.
        dec_X = Y.argmax(dim=2)
        py = dec_X.squeeze(dim=0).int().item()
        if py == tgt_vocab.eos:
            break
        predict_tokens.append(py)
    return ' '.join(tgt_vocab.to_tokens(predict_tokens))


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

    logging.info('引入了多层感知机注意力的 Seq2Seq 模型 ...')
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8,
                             num_hiddens=16, num_layers=2)
    # encoder.initialize()
    decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8,
                                      num_hiddens=16, num_layers=2)
    X = torch.zeros((4, 7), dtype=torch.long)
    print("batch size=4\nseq_length=7\nhidden dim=16\nnum_layers=2\n")
    print('encoder output size:', encoder(X)[0].size())
    print('encoder hidden size:', encoder(X)[1][0].size())
    print('encoder memory size:', encoder(X)[1][1].size())
    state = decoder.init_state(encoder(X), None)
    out, state = decoder(X, state)
    print(out.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)


if __name__ == '__main__':
    # test()

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.0
    batch_size, num_steps = 64, 10
    lr, num_epochs, ctx = 0.005, 500, try_gpu()

    src_vocab, tgt_vocab, train_iter = load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    model = EncoderDecoder(encoder, decoder)

    train_s2s_ch9(model, train_iter, lr, num_epochs, ctx)
    for sentence in ['Go .', 'Good Night !', "I'm OK .", 'I won !']:
        print(sentence + ' => ' + predict_s2s_ch9(
              model, sentence, src_vocab, tgt_vocab, num_steps, ctx))
