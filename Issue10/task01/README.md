第一天

线性回归；Softmax与分类模型、多层感知机

### 线性回归

LinearRegression.py 分别介绍了从零开始实现的线性回归和利用 PyTorch 实现的简洁版

从零开始也不是完全不使用 torch，只是没有直接使用 `torch.nn` 和 `torch.optim` 来创建模型并训练


### softmax

* softmax 函数

softmax 可以将神经元的输出映射到(0, 1) 区间，且和为一，这样就可以直接使用概率的方式来解释输出。

最大的特点是，求导计算很方便。尤其是跟交叉熵损失函数结合起来，求得的梯度很简洁。

即最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率。

* 交叉熵损失函数

交叉熵只关心对正确类别的预测概率，因为只要其值足够大，就可以确保分类结果正确。

* 对多维 Tensor 按维度操作 

`X.sum(dim=0, keepdim=True)` dim为0，按照相同的列求和，并在结果中保留列特征

`y_hat.gather(1, y.view(-1, 1))` 收集输入的特定维度指定位置的数值

这里的特定维度是指 dim 参数，如这个例子里，是对列进行操作。

三维 tensor 的元素选择

'''
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
'''

