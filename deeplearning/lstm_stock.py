#!D:/Code/python
# -*- coding: utf-8 -*-
# @File:lstm_stock.py
# @Software:PyCharm
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

df = pd.read_csv("D:/stock1.csv")
# 创建数据框
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Close'])
for i in range(0, len(data)):
    # new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# 创建训练集和验证集
dataset = new_data.values
train = dataset[0:200, :]
valid = dataset[200:, :]

# 将数据集转换为x_train和y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []

for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

print(len(x_train[0]))

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)
