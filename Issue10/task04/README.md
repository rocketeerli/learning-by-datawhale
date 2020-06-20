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
