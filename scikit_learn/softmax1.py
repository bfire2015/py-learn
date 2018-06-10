#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'Softmax回归算法与实战案例：葡萄酒质量分类'
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
from sklearn.preprocessing import MinMaxScaler

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

##拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

#读取数据
path1 = 'datas/winequality-red.csv'
df1 = pd.read_csv(path1, sep=';')
df1['type'] = 1 #设置数据类型为红葡萄酒

path2 = 'datas/winequality-white.csv'
df2 = pd.read_csv(path2, sep=';')
df2['type'] = 2 #设置数据类型为白葡萄酒

#合并两个df
df = pd.concat([df1, df2], axis=0)

## 自变量名称
names = ["fixed acidity","volatile acidity","citric acid",
         "residual sugar","chlorides","free sulfur dioxide",
         "total sulfur dioxide","density","pH","sulphates",
         "alcohol", "type"]
## 因变量名称
quality = "quality"

## 显示
# print(df.head(5))

##异常数据处理
new_df = df.replace('?', np.nan)
datas= new_df.dropna(how = 'any')
## 只要有列为空，就进行删除操作
print ("原始数据条数:%d；异常数据处理后数据条数:%d；异常数据条数:%d" % (len(df), len(datas), len(df) - len(datas)))

## 提取自变量和因变量
x = datas[names]
y = datas[quality]

##数据分割
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25, random_state=0)

print('训练数据条数：%d;数据特征个数:%d;测试数据条数：%d'%(x_train.shape[0], x_train.shape[1], x_test.shape[0]))

#2\数据格式化将数据缩放到[0,1]之前是StandardScaler
ss = MinMaxScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

##查看y值的范围和数理统计
print(y_train.value_counts())

##3\模型构建及训练
## multi_class: 分类方式参数；参数可选: ovr(默认)、multinomial；这两种方式在二元分类问题中，效果是一样的；在多元分类问题中，效果不一样
### ovr: one-vs-rest， 对于多元分类的问题，先将其看做二元分类，分类完成后，再迭代对其中一类继续进行二元分类 里面会有多个模型
### multinomial: many-vs-many（MVM）,即Softmax分类效果 只有一个模型
## class_weight: 特征权重参数

### Softmax算法相对于Logistic算法来讲，在sklearn中体现的代码形式来讲，主要只是参数的不同
# Softmax k个θ向量并不是表示有k个模型 底层只有一个模型 在模型中训练的是θ矩阵而非向量

lr = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-5, 1, 100), multi_class='multinomial', penalty='l2', solver='lbfgs')

lr.fit(x_train, y_train)

#4\模型效果获取
r = lr.score(x_train, y_train)
print('r值=',r)
print('特征稀疏化比率：%.2f%%'%(np.mean(lr.coef_.ravel() == 0) * 100))
print('参数', lr.coef_)
print('截距', lr.intercept_)

print('概率', lr.predict_proba(x_train))

#获取概率函数返回概率值
print('概率有多少',lr.predict_proba(x_test).shape)

#数据预测
#1\预测数据格式化
#2、结果数据预测
y_predict = lr.predict(x_test)

#3\图表展示
x_len = range(len(x_test))
plt.figure(figsize=(14, 7), facecolor='w')
plt.ylim(-1, 11)
plt.plot(x_len, y_test, 'ro', markersize = 8, zorder=3, label='真实值')
plt.plot(x_len, y_predict, 'go', markersize = 12, zorder=2, label='预测值准确率,$r^2$=%.3f'%lr.score(x_train, y_train))
plt.legend(loc = 'upper left')
plt.xlabel('数据编号ID',fontsize=18)
plt.ylabel('葡萄酒质量层级',fontsize=18)
plt.title('葡萄酒质量预测分析',fontsize=20)
plt.show()

