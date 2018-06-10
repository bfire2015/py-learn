#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '34、使用决策树的可视化工具画出树结构_笔记 tree6'
__author__ = 'BfireLai'
__mtime__ = '2018/5/22'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model.coordinate_descent import ConvergenceWarning


def notEmpty(s):
	return s != ''

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore', category=FutureWarning)

names = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
path = "datas/boston_housing.data"
## 由于数据文件格式不统一，所以读取的时候，先按照一行一个字段属性读取数据，然后再按照每行数据进行处理

fd = pd.read_csv(path, header=None)
data = np.empty((len(fd), 14))
for i,d in enumerate(fd.values):
	d = map(float, filter(notEmpty, d[0].split(' ')))
	data[i] = list(d)


x,y = np.split(data, (13,), axis=1)
y = y.ravel()

print('样本数据量%d,特征个数：%d'%x.shape)
print('target样本数据量%d'%y.shape[0])

#数据的分割，
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=14)
x_train, x_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))

#标准化
ss = MinMaxScaler()

x_train = ss.fit_transform(x_train, y_train)
x_test = ss.transform(x_test)

print ("原始数据各个特征属性的调整最小值:",ss.min_)
print ("原始数据各个特征属性的缩放数据值:",ss.scale_)

#构建模型（回归）
model = DecisionTreeRegressor(criterion='mse', max_depth=5, random_state=8)

#模型训练
model.fit(x_train, y_train)

#模型预测
y_test_hat = model.predict(x_test)

#评估模型
score = model.score(x_test, y_test)
print('score', score)

#直接生成图片
from sklearn import tree
import  pydotplus
dot_data = tree.export_graphviz(model, out_file=None, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('./datas/fangjia.png')