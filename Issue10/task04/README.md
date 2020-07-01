第四天

文本预处理；语言模型；循环神经网络基础

### 文本预处理

文本是一类序列数据，一篇文章可以看作是字符或单词的序列。

* 流程

1. 读入文本
2. 分词
3. 建立字典，将每个词映射到一个唯一的索引（index）
4. 将文本从词的序列转换为索引的序列，方便输入模型

* 分词的工具库

**spaCy**：

	import spacy
	nlp = spacy.load('en_core_web_sm')
	doc = nlp(text)
	print([token.text for token in doc])

**NLTK**：

	from nltk.tokenize import word_tokenize
	from nltk import data
	data.path.append('/home/kesci/input/nltk_data3784/nltk_data')
	print(word_tokenize(text))

### 语言模型

语言模型的参数就是词的概率以及给定前几个词情况下的条件概率。

* n 元语法

随着序列长度增加，计算和存储多个词共同出现的概率的复杂度会呈指数级增加。

n 元语法是通过马尔可夫假设简化模型。马尔科夫假设是指一个词的出现只与前面 n 个词相关，即 n 阶马尔可夫链（Markov chain of order n）。

n 元语法（n-grams）是基于 n-1 阶马尔可夫链的概率语言模型。

当 n 分别为1、2和3时，我们将其分别称作一元语法（unigram）、二元语法（bigram）和三元语法（trigram）。

* n 元语法缺陷

1. 参数空间过大

2. 数据稀疏

#### 时序数据的采样

* 随机采样

在随机采样中，每个样本是原始序列上任意截取的一段序列，相邻的两个随机小批量在原始序列上的位置不一定相毗邻。

* 相邻采样

在相邻采样中，相邻的两个随机小批量在原始序列上的位置相毗邻。

### 循环神经网络基础

* one hot

采用 one-hot 编码将字符表示成向量。

#### RNN

相比于前馈神经网络多了一个时步（timestep）

* RNN 的 batch

主流的优化算法是基于mini-batch的，也就是训练完一个batch后更新一次权重，这是训练完整个epoch后更新权重和训练单个样本后更新权重两者的折衷.

这一折衷通常情况下能够加速收敛进程，从而加快训练进程。

**因为mini-batch内权重不变，所以可以很方便地利用GPU的并行功能进一步加速训练。**

注意：**前馈网络的一个batch是一“条”，RNN的一个batch是一“片”**

参考：[LSTM的一个batch到底是怎么进入神经网络的？](https://www.zhihu.com/question/286310691)

* 裁剪梯度

裁剪梯度（clip gradient）是一种应对梯度爆炸的方法。


