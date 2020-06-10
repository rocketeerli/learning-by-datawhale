# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import os
from Overfitting import show_data
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')
torch.set_default_tensor_type(torch.FloatTensor)


class Dataset():
    def __init__(self, root='./data/HousePrices'):
        self.root = root
        self.test_data, self.train_data = self._read_data()

    def _read_data(self):
        test_data = pd.read_csv(os.path.join(self.root, 'test.csv'))
        train_data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        # logging.info(f'data type: {type(test_data)}')
        # logging.info(f'data shape: {train_data.shape}')
        # logging.info(f'data head: \n{train_data.head()}')
        # logging.info(f'data 0-4: \n{train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]}')
        return test_data, train_data

    def preprocessed_data(self):
        ''' 预处理数据 '''
        # 对连续数值的特征做标准化
        all_features = pd.concat((self.train_data.iloc[:, 1:-1],
                                 self.test_data.iloc[:, 1:]))
        is_numeric = all_features.dtypes != 'object'
        numeric_features = all_features.dtypes[is_numeric].index
        all_features[numeric_features] = all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / x.std()
        )
        all_features[numeric_features] = all_features[numeric_features].fillna(0)
        # 将离散数值转成指示特征（one hot 编码）
        all_features = pd.get_dummies(all_features, dummy_na=True)
        # logging.info(f'shape of all_features: {all_features.shape}')
        # 通过 values 属性得到 numpy 格式的数据，并转成 tensor
        n_train = self.train_data.shape[0]
        train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
        test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
        train_labels = torch.tensor(self.train_data.SalePrice.values, dtype=torch.float).view(-1, 1)
        return train_features, test_features, train_labels


class PricePrediction():
    def __init__(self, inputs, lr, wd, epochs, batch_size):
        self.inputs = inputs
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = self._get_net()
        # 均方误差损失
        self.loss = nn.MSELoss()
        # 使用 Adam 优化
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr,
                                          weight_decay=wd)

    def _get_net(self):
        ''' 定义模型 '''
        net = nn.Linear(self.inputs, 1)
        for param in net.parameters():
            nn.init.normal_(param, mean=0, std=0.01)
        return net

    def log_rmse(self, features, labels):
        ''' 对数均方根误差 '''
        with torch.no_grad():
            # 截断，将小于 1 的设成 1
            clipped_preds = torch.max(self.net(features), torch.tensor(1.0))
            loss = 2 * self.loss(clipped_preds.log(), labels.log())
            rmse = torch.sqrt(loss)
        return rmse.item()

    def train(self, train_features, train_labels, test_features, test_labels):
        train_ls, test_ls = [], []
        dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        train_iter = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            for X, y in train_iter:
                loss = self.loss(self.net(X.float()), y.float())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_loss = self.log_rmse(train_features, train_labels)
            train_ls.append(train_loss)
            if test_labels is not None:
                test_loss = self.log_rmse(test_features, test_labels)
                logging.info(f'epoch {epoch+1} \t train loss: {train_loss} \t test loss: {test_loss}')
                test_ls.append(test_loss)            
        return train_ls, test_ls

    def get_k_fold_data(self, k, i, X, y):
        # 返回第 i 折交叉验证时所需要的训练集和验证集
        assert k > 1
        fold_size = X.shape[0] // k
        X_train, y_train = None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part, y_part = X[idx, :], y[idx]
            if j == i:
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat((X_train, X_part), dim=0)
                y_train = torch.cat((y_train, y_part), dim=0)
        return X_train, y_train, X_valid, y_valid

    def k_fold(self, k, X_train, y_train):
        train_l_sum, valid_l_sum = 0, 0
        for i in range(k):
            data = self.get_k_fold_data(k, i, X_train, y_train)
            train_ls, valid_ls = self.train(*data)
            train_l_sum += train_ls[-1]
            valid_l_sum += valid_ls[-1]
            plt.title(f'{i+1} train and valid')
            show_data(range(1, self.epochs + 1), train_ls, 'epochs', 'rmse',
                      range(1, self.epochs + 1), valid_ls, ['train', 'valid'])
        logging.info('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
        return train_l_sum / k, valid_l_sum / k

    def train_and_pred(self, train_features, test_features, train_labels, test_data):
        train_ls, _ = self.train(train_features, train_labels, None, None)
        show_data(range(1, self.epochs + 1), train_ls, 'epochs', 'rmse')
        print('train rmse %f' % train_ls[-1])
        preds = self.net(test_features).detach().numpy()
        test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
        submission.to_csv('./submission.csv', index=False)


if __name__ == '__main__':
    dataset = Dataset()
    train_features, test_features, train_labels = dataset.preprocessed_data()
    K, epochs, lr, wd, batch_size = 5, 100, 5, 0, 64
    price_prediction = PricePrediction(train_features.shape[-1], lr, wd, epochs, batch_size)
    train_l, valid_l = price_prediction.k_fold(K, train_features, train_labels)
    logging.info('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (K, train_l, valid_l))
    price_prediction.train_and_pred(train_features, test_features, train_labels, dataset.test_data)
