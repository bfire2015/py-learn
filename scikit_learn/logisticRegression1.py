#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '逻辑回归代码实战案例：乳腺癌预测'
__author__ = 'BfireLai'
__mtime__ = '2018/5/17'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

##拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

##数据读取并处理异常数据
path = 'datas/breast-cancer-wisconsin.data'
names = ['id', 'clump thickness', 'uniformity of cell size', 'uniformity of cell shape', 'marginal adhesion',
		 'single epithelial cell size', 'bare nuclei', 'bland chromatin', 'normal nucleoli', 'mitoses', 'class']

df = pd.read_csv(path, header=None, names=names)

datas = df.replace('?', np.nan).dropna(how = 'any')#只要有列为空，就进行删除操作
print(datas.head(5))

#1\数据提取以及数据分隔
#提取
X = datas[names[1:10]]
Y = datas[names[10]]

#分隔
x_train, x_test, y_train,y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

#2\数据格式化（归一化）
ss = StandardScaler()
x_train = ss.fit_transform(x_train)##训练模型及归一化数据

#3\模型构建及训练
## penalty: 过拟合解决参数过大,l1或者l2
## solver: 参数优化方式
### 当penalty为l1的时候，参数只能是：liblinear(坐标轴下降法)后面会讲 l1正则的导数不是连续的就没法求解# ；
### nlbfgs和cg都是关于目标函数的二阶泰勒展开
### 当penalty为l2的时候，参数可以是：lbfgs(拟牛顿法)、newton-cg(牛顿法变种)，seg(minibatch)
# 维度<10000时，lbfgs法比较好，   维度>10000时， cg法比较好，显卡计算的时候，lbfgs和cg都比seg快
## multi_class: 分类方式参数；参数可选: ovr(默认)、multinomial；这两种方式在二元分类问题中，效果是一样的；在多元分类问题中，效果不一样
### ovr: one-vs-rest， 对于多元分类的问题，先将其看做二元分类，分类完成后，再迭代对其中一类继续进行二元分类 里面会有多个模型
### multinomial: many-vs-many（MVM）,即Softmax分类效果 只有一个模型
## class_weight: 特征权重参数

### TODO: Logistic回归是一种分类算法，不能应用于回归中(也即是说对于传入模型的y值来讲，不能是float类型，必须是int类型)

lr = LogisticRegressionCV(multi_class='ovr', fit_intercept=True, Cs=np.logspace(-2, 2, 20),cv=2, penalty='l2', solver='lbfgs', tol=0.01)
re = lr.fit(x_train, y_train)

#4\模型效果获取
r = re.score(x_train, y_train)
print('r^2=',r)
print('稀疏化特征比率%.2f%%'%(np.mean(lr.coef_.ravel() == 0)*1000))
print('参数：', re.coef_)
print('截距',re.intercept_)
#获取sigmoid函数返回概率值 p 概率模型
print(re.predict_proba(x_test))

#数据预测
## 预测数据格式化
x_test = ss.transform(x_test)#使用模型进行归一化
##结果数据预测
y_predict = re.predict(x_test)

#图表展示
x_len = range(len(x_test))
plt.figure(figsize=(14,7), facecolor='w')
plt.ylim(0,6)
plt.plot(x_len, y_test, 'ro', markersize=8, zorder=3, label='真实值')
plt.plot(x_len, y_predict, 'go', markersize=14, zorder=2, label='预测值,$r^2$=%.3f'%re.score(x_test, y_test))
plt.legend(loc= 'upper left')
plt.xlabel('数据编号', fontsize=18)
plt.ylabel('乳腺癌类型', fontsize=18)
plt.title('logistic回归算法对乳腺癌数据进行分类', fontsize=20)
plt.show()