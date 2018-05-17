#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'ML分类问题综合实战案例：信贷审批与鸢尾花分类2'
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
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn import metrics

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

##拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

path = 'datas/iris.data'
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'cla']
df = pd.read_csv(path, header=None, names=names)
print(df['cla'].value_counts())
print(df.head())
print(df['cla'].value_counts())

def parseRecord(record):
	result = []
	r = zip(names, record)
	for name,v in r:
		if name == 'cla':
			if v == 'Iris-setosa':
				result.append(1)
			elif v == 'Iris-versicolor':
				result.append(2)
			elif v == 'Iris-virginica':
				result.append(3)
			else:
				result.append(np.nan)
		else:
			result.append(float(v))
	return result


### 1. 数据转换为数字以及分割
## 数据转换
datas = df.apply(lambda r:parseRecord(r), axis=1)
##异常数据删除
datas = datas.dropna(how='any')
##数据分割
x = datas[names[0:-1]]
y = datas[names[-1]]

##数据抽样（训练数据和测试数据分割）
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

print('原始数据条数：%d;训练数据条数：%d,特征个数：%d;测试样本条数：%d'%(len(x), len(x_train), x_train.shape[1], x_test.shape[0]))

##2\数据标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

##3\特征选择
##4、降维处理
##5、模型构建

lr = LogisticRegressionCV(Cs=np.logspace(-4, 1, 50), cv=3, fit_intercept=True, penalty='l2', solver='lbfgs', tol=0.01, multi_class='multinomial')
#solver：‘newton-cg’,'lbfgs','liblinear','sag'  default:liblinear
#'sag'=mini-batch
#'multi_clss':
lr.fit(x_train, y_train)

## 6. 模型效果输出
## 将正确的数据转换为矩阵形式(每个类别使用向量的形式来表述)
# 其中的一种哑编码的形式
print(y_test)
y_test_hot = label_binarize(y_test, classes=(1,2,3))
print(y_test_hot)

## 得到预测的损失值
lr_y_score = lr.decision_function(x_test)
##计算roc 的值
lr_fpr,lr_tpr,lr_threasholds = metrics.roc_curve(y_test_hot.ravel(), lr_y_score.ravel())
##threasholds
##计算auc的值
lr_auc = metrics.auc(lr_fpr, lr_tpr)
print('logistic算法r^2',lr.score(x_train, y_train))
print('logistic算法auc值',lr_auc)

##7.模型预测
lr_y_predict = lr.predict(x_test)

##knn算法实现
#1、模型构建
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

##2\模型效果输出
##将正确的数据转换为矩阵形式
y_test_hot = label_binarize(y_test, classes=(1, 2, 3))
##得到预测属于某个类别的概率值
knn_y_score = knn.predict_proba(x_test)
##计算roc的值
knn_fpr, knn_tpr, knn_threasholds = metrics.roc_curve(y_test_hot.ravel(), knn_y_score.ravel())
##计算auc的值
knn_auc = metrics.auc(knn_fpr, knn_tpr)
print('knn算法r值 ', knn.score(x_train, y_train))
print('knn算法auc值', knn_auc)

##3\模型观测
knn_y_predict = knn.predict(x_test)

##画图1\roc 曲线画图
plt.figure(figsize=(8,6), facecolor='w')
plt.plot(lr_fpr, lr_tpr, c='r', lw=2, label='logistic算法auc=%.3f'%lr_auc)
plt.plot(knn_fpr, knn_tpr, c='g', lw=2, label='knn算法auc=%.3f'%knn_auc)
plt.plot((0,1), (0,1), c='#a0a0a0', lw=2, ls='--')
plt.xlim(-0.01, 1.02)#设置x轴最大和最小值
plt.ylim(-0.01, 1.02)#设置y轴最大和最小值
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('false positive rate(fpr)', fontsize=16)
plt.ylabel('true positive rate(tpr)', fontsize=16)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title('鸢尾花数据Logistic和KNN算法的ROC/AUC', fontsize=18)
plt.show()

## 画图2：预测结果画图
x_test_len = range(len(x_test))
plt.figure(figsize=(12, 9), facecolor='w')
plt.ylim(0.5,3.5)
plt.plot(x_test_len, y_test, 'ro',markersize = 6, zorder=3, label=u'真实值')
plt.plot(x_test_len, lr_y_predict, 'go', markersize = 10, zorder=2, label='Logis算法（测试集）,$R^2$=%.3f' % lr.score(x_test, y_test))
plt.plot(x_test_len, knn_y_predict, 'yo', markersize = 16, zorder=1, label='KNN算法（测试集）,$R^2$=%.3f' % knn.score(x_test, y_test))
plt.legend(loc = 'lower right')
plt.xlabel('数据编号', fontsize=18)
plt.ylabel('种类', fontsize=18)
plt.title('鸢尾花数据分类', fontsize=20)
plt.show()
