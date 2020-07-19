# -*- coding:utf-8 -*-
import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data


def read_data():
    with open('./data/ptb_train/ptb.train.txt', 'r') as f:
        lines = f.readlines()  # 该数据集中句子以换行符为分割
        raw_dataset = [st.split() for st in lines]  # st是sentence的缩写，单词以空格为分割
    print('# sentences: %d' % len(raw_dataset))
    # 对于数据集的前3个句子，打印每个句子的词数和前5个词
    # 句尾符为 '' ，生僻词全用 '' 表示，数字则被替换成了 'N'
    for st in raw_dataset[:3]:
        print('# tokens:', len(st), st[:5])

    counter = collections.Counter([tk for st in raw_dataset for tk in st])  # tk是token的缩写
    # items()  # 转化成(元素，计数值)组成的列表
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))  # 只保留在数据集中至少出现5次的词
    idx_to_token = [tk for tk, _ in counter.items()]
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
    dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
               for st in raw_dataset]  # raw_dataset中的单词在这一步被转换为对应的idx
    num_tokens = sum([len(st) for st in dataset])
    print('# tokens: %d' % num_tokens)
    return counter, idx_to_token, token_to_idx, dataset, num_tokens


def discard(idx):
    '''
    @params:
        idx: 单词的下标
    @return: True/False 表示是否丢弃该单词
    '''
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)


def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))


def get_centers_and_contexts(dataset, max_window_size):
    '''
    @params:
        dataset: 数据集为句子的集合，每个句子则为单词的集合，此时单词已经被转换为相应数字下标
        max_window_size: 背景词的词窗大小的最大值
    @return:
        centers: 中心词的集合
        contexts: 背景词窗的集合，与中心词对应，每个背景词窗则为背景词的集合
    '''
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size) # 随机选取背景词窗大小
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    ''' Skip-Gram 模型的前向计算
    @params:
        center: 中心词下标，形状为 (n, 1) 的整数张量
        contexts_and_negatives: 背景词和噪音词下标，形状为 (n, m) 的整数张量
        embed_v: 中心词的 embedding 层
        embed_u: 背景词的 embedding 层
    @return:
        pred: 中心词与背景词（或噪音词）的内积，之后可用于计算概率 p(w_o|w_c)
    '''
    v = embed_v(center)  # shape of (n, 1, d)
    u = embed_u(contexts_and_negatives)  # shape of (n, m, d)
    pred = torch.bmm(v, u.permute(0, 2, 1))  # bmm((n, 1, d), (n, d, m)) => shape of (n, 1, m)
    return pred


def get_negatives(all_contexts, sampling_weights, K):
    '''
    @params:
        all_contexts: [[w_o1, w_o2, ...], [...], ... ]
        sampling_weights: 每个单词的噪声词采样概率
        K: 随机采样个数
    @return:
        all_negatives: [[w_n1, w_n2, ...], [...], ...]
    '''
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)


def batchify(data):
    '''
    用作DataLoader的参数collate_fn
    @params:
        data: 长为batch_size的列表，列表中的每个元素都是__getitem__得到的结果
    @outputs:
        batch: 批量化后得到 (centers, contexts_negatives, masks, labels) 元组
            centers: 中心词下标，形状为 (n, 1) 的整数张量
            contexts_negatives: 背景词和噪声词的下标，形状为 (n, m) 的整数张量
            masks: 与补齐相对应的掩码，形状为 (n, m) 的0/1整数张量
            labels: 指示中心词的标签，形状为 (n, m) 的0/1整数张量
    '''
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]  # 使用掩码变量mask来避免填充项对损失函数计算的影响
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    batch = (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
             torch.tensor(masks), torch.tensor(labels))
    return batch


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        '''
        @params:
            inputs: 经过sigmoid层后为预测D=1的概率
            targets: 0/1向量，1代表背景词，0代表噪音词
        @return:
            res: 平均到每个label的loss
        '''
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        res = res.sum(dim=1) / mask.float().sum(dim=1)
        return res


def sigmd(x):
    return - math.log(1 / (1 + math.exp(-x)))


def test_pytorch():
    # Embedding 层
    embed = nn.Embedding(num_embeddings=10, embedding_dim=4)
    print(embed.weight)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    print(embed(x))
    # 批量乘法
    X = torch.ones((2, 1, 4))
    Y = torch.ones((2, 4, 6))
    print(torch.bmm(X, Y).shape)


def train(net, lr, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train on", device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])

            l = loss(pred.view(label.shape), label, mask).mean() # 一个batch的平均loss
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))


def get_similar_tokens(query_token, k, embed):
    '''
    @params:
        query_token: 给定的词语
        k: 近义词的个数
        embed: 预训练词向量
    '''
    W = embed.weight.data
    x = W[token_to_idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i], (idx_to_token[i])))


if __name__ == '__main__':
    counter, idx_to_token, token_to_idx, dataset, num_tokens = read_data()

    # 二次采样
    subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
    print('# tokens: %d' % sum([len(st) for st in subsampled_dataset]))
    print(compare_counts('the'))
    print(compare_counts('join'))

    # 提取中心词和背景词
    all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)
    tiny_dataset = [list(range(7)), list(range(7, 10))]
    print('dataset', tiny_dataset)
    for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
        print('center', center, 'has contexts', context)

    # PyTorch API 测试
    # test_pytorch()

    # 负采样近似
    sampling_weights = [counter[w]**0.75 for w in idx_to_token]
    all_negatives = get_negatives(all_contexts, sampling_weights, 5)

    # 批量读取数据
    batch_size = 512
    num_workers = 0 if sys.platform.startswith('win32') else 4
    dataset = MyDataset(all_centers, all_contexts, all_negatives)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True,
                                collate_fn=batchify,
                                num_workers=num_workers)
    # for batch in data_iter:
    #     for name, data in zip(['centers', 'contexts_negatives', 'masks',
    #                            'labels'], batch):
    #         print(name, 'shape:', data.shape)
    #     break

    loss = SigmoidBinaryCrossEntropyLoss()
    pred = torch.tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
    label = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]]) # 标签变量label中的1和0分别代表背景词和噪声词
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])  # 掩码变量
    print(loss(pred, label, mask))

    print('%.4f' % ((sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-2)) / 4)) # 注意1-sigmoid(x) = sigmoid(-x)
    print('%.4f' % ((sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))

    embed_size = 100
    net = nn.Sequential(nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size))

    # 训练
    train(net, 0.01, 5)

    # 测试模型
    get_similar_tokens('chip', 3, net[0])
