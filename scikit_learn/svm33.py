#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '69、SVM算法4个实战综合案例_笔记3'
__author__ = 'BfireLai'
__mtime__ = '2018/7/25'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn

from sklearn.svm import SVR #对比svc 是svm的回归形式
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

def notEmpty(s):
	return s != ''

##加载数据
#names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
path = 'datas/boston_housing.data'
fd = pd.read_csv(path, header=None)
data = np.empty((len(fd), 14))
for i,d in enumerate(fd.values):
	d = map(float, filter(notEmpty, d[0].split(' ')))
	data[i] = list(d)


##分割数据
x,y= np.split(data, (13,), axis=1)
y = y.ravel() #转换格式
print('样本数据量%d,特征个数：%d'%x.shape)
print('target样本数据量%d'%y.shape[0])

#数据分割
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=28)

#模型构建，
parameters = {
	'kernel': ['linear', 'rbf'],
	'C': [0.1, 0.5, 0.9, 1, 5],
	'gamma': [0.001, 0.01, 0.1, 1]
}
model = GridSearchCV(SVR(), param_grid=parameters, cv=3)
model.fit(x_train, y_train)

##获取最优参数
print('最优参数列表',model.best_params_)
print('最优模型',model.best_estimator_)
print('最优效率', model.best_score_)

##模型效果输出
print('训练集准确率%.2f%%'%(model.score(x_train, y_train)*100))
print('测试集准确率%.2f%%'%(model.score(x_test, y_test)*100))

##画图
colors = ['g-', 'b-']
ln_x_test = range(len(x_test))
y_predict = model.predict(x_test)

plt.figure(figsize=(16, 8), facecolor='w')
plt.plot(ln_x_test, y_test, 'r-', lw=2, label='真实值')
plt.plot(ln_x_test, y_predict, 'g-', lw=3, label='svr算法估值，$r^2$=%.3f'%(model.best_score_))

##图形显示
plt.legend(loc='upper left')
plt.grid(True)
plt.title('波士顿房屋价格预测(svr)')
plt.xlim(0, 101)
plt.show()