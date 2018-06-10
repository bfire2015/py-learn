#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '42、AdaBoost算法的实战代码案例与总结_笔记 2'
__author__ = 'BfireLai'
__mtime__ = '2018/5/22'
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score #计算roc 和auc
from sklearn.tree import DecisionTreeClassifier

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

##创建模拟数据
x,y = make_gaussian_quantiles(n_samples=13000,n_features=10, n_classes=3, random_state=1)

n_split = 3000
x_train,x_test = x[:n_split], x[n_split:]
y_train,y_test = y[:n_split], y[n_split:]

#建立两个模型， algorithm算法不同
bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1, algorithm='SAMME.R')

#样本个数
bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1, algorithm='SAMME')

bdt_real.fit(x_train, y_train)
bdt_discrete.fit(x_train, y_train)

# 获得预测的准确率，accuracy_score，是单个分类器的准确率。
# 预测的误差率estimator_errors_
real_test_errors = [] #第一个模型每一个分类器的误差率
discrete_test_errors = [] #第二个模型每一个分类器的误差率

# staged_predict 分阶段的预测 每训练出一个模型 会融合成一个强学习器
for real_test_predict, discrete_train_predict in zip(bdt_real.staged_predict(x_test), bdt_discrete.staged_predict(x_test)):
	real_test_errors.append(1.-accuracy_score(real_test_predict, y_test))
	discrete_test_errors.append(1.-accuracy_score(discrete_train_predict, y_test))

n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)

discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]

#获得每个子模型的权重alpha
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]
real_estimator_weights = bdt_real.estimator_weights_[:n_trees_real]
print('SAMME权重：', discrete_estimator_weights)
print('SAMME权重(个数)：', len(discrete_estimator_weights))
print('SAMME.R权重(全是1)：', real_estimator_weights)
print('SAMME.R权重（全是1）（个数）：', len(real_estimator_weights))

#开始画图
plt.figure(figsize=(15, 5), facecolor='w')
#子图1 1行3列
plt.subplot(131)

plt.plot(range(1, n_trees_discrete + 1), discrete_test_errors, c='g', label='SAMME')
plt.plot(range(1, n_trees_real + 1), real_test_errors, c='r', linestyle='dashed', label='SAMME.r')

plt.legend()
plt.ylim(0.18, 0.62)
plt.ylabel('测试数据的预测错误率')
plt.xlabel('弱分类器数目')
plt.title('区分迭代速度')

#子图2 1行3列
plt.subplot(132)
plt.plot(range(1, n_trees_discrete +1), discrete_estimator_errors, 'b', label='samme', alpha=.5)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors, 'r', label='SAMME.r', alpha=.5)
plt.legend()
plt.ylabel('每个子模型实际错误率')
plt.xlabel('弱分类器数目')
plt.title('每个子模型的错误率')
plt.ylim((.2,max(real_estimator_errors.max(), discrete_estimator_errors.max()) *1.2))
plt.xlim((-20, len(bdt_discrete) + 20))

#子图3 1行3列
plt.subplot(133)
plt.plot(range(1, n_trees_discrete +1), discrete_estimator_weights, 'b', label='samme')
plt.plot(range(1, n_trees_real + 1), real_estimator_weights, 'r', label='SAMME.r')
plt.legend()
plt.ylabel('权重')
plt.xlabel('弱分类器编号')
plt.title('每个子模型的权重')
plt.ylim((0, discrete_estimator_errors.max() *1.2))
plt.xlim((-20, n_trees_discrete + 20))

plt.subplots_adjust(wspace=0.25)
plt.show()







