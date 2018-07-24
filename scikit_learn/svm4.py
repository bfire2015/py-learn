#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '68、SVM算法5个代码案例_加深理解SVM算法原理_笔记4'
__author__ = 'BfireLai'
__mtime__ = '2018/7/24'
"""

import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

from sklearn.svm import SVC #svm导入
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ChangedBehaviorWarning

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore', category=ChangedBehaviorWarning)

##读取数据
## 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
path = './datas/iris.data' #数据源
data = pd.read_csv(path, header=None)
x, y = data[list(range(4))], data[4]

print(y.value_counts())
print(pd.Categorical(y).categories)
y = pd.Categorical(y).codes #把文本数据进行编码 比如abc编码为012可以通过pd.Categorical(y).categories获取index对应的原始值
print(y)
print(x.head())

x = x[[0, 1]]
print(x.head())

#数据分割
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=28, test_size=0.6)

## svm.SVC API说明：
# 功能：使用SVM分类器进行模型构建
# 参数说明：
# C: 误差项的惩罚系数，默认为1.0；一般为大于0的一个数字，C越大表示在训练过程中对于总误差的关注度越高，也就是说当C越大的时候，对于训练集的表现会越好，
# 但是有可能引发过度拟合的问题(overfiting)
# kernel：指定SVM内部核函数的类型，可选值：linear、poly、rbf、sigmoid、precomputed(基本不用，有前提要求，要求特征属性数目和样本数目一样)；默认是rbf；
# degree：当使用多项式函数作为svm内部的函数的时候，给定多项式的项数，默认为3
# gamma：当SVM内部使用poly、rbf、sigmoid的时候，核函数的系数值，当默认值为auto的时候，实际系数为1/n_features
# coef0: 当核函数为poly或者sigmoid的时候，给定的独立系数，默认为0
# probability：是否启用概率估计，默认不启动，不太建议启动
# shrinking：是否开启收缩启发式计算，默认为True
# tol: 模型构建收敛参数，当模型的的误差变化率小于该值的时候，结束模型构建过程，默认值:1e-3
# cache_size：在模型构建过程中，缓存数据的最大内存大小，默认为空，单位MB
# class_weight：给定各个类别的权重，默认为空
# max_iter：最大迭代次数，默认-1表示不限制
# decision_function_shape: 决策函数，可选值：ovo和ovr，默认为None；推荐使用ovr；（1.7以上版本才有）
# '''

## 数据SVM分类器构建
svm1 = SVC(C=0.1, kernel='rbf')
svm2 = SVC(C=1, kernel='rbf')
svm3 = SVC(C=10, kernel='rbf')
svm4 = SVC(C=100, kernel='rbf')
svm5 = SVC(C=500, kernel='rbf')
svm6 = SVC(C=100000, kernel='rbf')

#clf = svm.SVC(C=1, kernel='rbf', gamma=0.1)
#gamma值越大，训练集的拟合就越好，但是会造成过拟合，导致测试集拟合变差
#gamma值越小，模型的泛化能力越好，训练集和测试集的拟合相近，但是会导致训练集出现欠拟合问题，
#从而，准确率变低，导致测试集准确率也变低。


#C越大，泛化能力越差，会出现过拟合的问题
#C越小，泛化能力越好，但是容易出现欠拟合的问题

## 模型训练
t0 = time.time()
svm1.fit(x_train, y_train)
t1 = time.time()
svm2.fit(x_train, y_train)
t2 = time.time()
svm3.fit(x_train, y_train)
t3 = time.time()
svm4.fit(x_train, y_train)
t4 = time.time()
svm5.fit(x_train, y_train)
t5 = time.time()
svm6.fit(x_train, y_train)
t6 = time.time()

##效果评估
svm1_score1 = accuracy_score(y_train, svm1.predict(x_train))
svm1_score2 = accuracy_score(y_test, svm1.predict(x_test))

svm2_score1 = accuracy_score(y_train, svm2.predict(x_train))
svm2_score2 = accuracy_score(y_test, svm2.predict(x_test))

svm3_score1 = accuracy_score(y_train, svm3.predict(x_train))
svm3_score2 = accuracy_score(y_test, svm3.predict(x_test))

svm4_score1 = accuracy_score(y_train, svm4.predict(x_train))
svm4_score2 = accuracy_score(y_test, svm4.predict(x_test))

svm5_score1 = accuracy_score(y_train, svm5.predict(x_train))
svm5_score2 = accuracy_score(y_test, svm5.predict(x_test))

svm6_score1 = accuracy_score(y_train, svm6.predict(x_train))
svm6_score2 = accuracy_score(y_test, svm6.predict(x_test))


#画图
x_tmp = [0, 1, 2, 3, 4, 5]
t_score = [t1 - t0, t2- t1,  t3- t2, t4-t3, t5-t4, t6-t5]
y_score1 = [svm1_score1, svm2_score1, svm3_score1, svm4_score1, svm5_score1, svm6_score1]
y_score2 = [svm1_score2, svm2_score2, svm3_score2, svm4_score2, svm5_score2, svm6_score2]

plt.figure(facecolor='w', figsize=(12, 6))

plt.subplot(121)
plt.plot(x_tmp, y_score1, 'r-', lw=2, label='训练集准确率')
plt.plot(x_tmp, y_score2, 'g-', lw=2, label='测试集准确率')
plt.xlim(-0.3, 3.3)
plt.ylim(np.min((np.min(y_score1), np.min(y_score2)))*0.9, np.max((np.max(y_score1), np.max(y_score2)))*1.1)
plt.legend(loc='lower right')
plt.title('模型预测准确率比较', fontsize=13)
plt.xticks(x_tmp, ['c-0.1', 'c-1', 'c-10', 'c-100', 'c-500', 'c-10000'], rotation=0)
plt.grid(b=True)

plt.subplot(122)
plt.plot(x_tmp, t_score, 'b-', lw=2, label='模型训练时间')
plt.xlim(-0.3, 3.3)
plt.title('模型训练耗时', fontsize=13)
plt.xticks(x_tmp, ['c-0.1', 'c-1', 'c-10', 'c-100', 'c-500', 'c-10000'], rotation=0)
plt.grid(b=True)


plt.suptitle('鸢尾花数据SVM分类器不同惩罚项系数模型比较', fontsize=18, color='r')
plt.show()


#画图比较
N = 500
x1_min, x2_min = x.min()
x1_max, x2_max = x.max()

t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
x1, x2 = np.meshgrid(t1, t2)
grid_show = np.dstack((x1.flat, x2.flat))[0]#测试点

##获取各个不同算法的测试值
svm1_grid_hat = svm1.predict(grid_show)  #预测分类值
svm1_grid_hat = svm1_grid_hat.reshape(x1.shape)#使之与输入的形状相同

svm2_grid_hat = svm2.predict(grid_show)
svm2_grid_hat = svm2_grid_hat.reshape(x1.shape)

svm3_grid_hat = svm3.predict(grid_show)
svm3_grid_hat = svm3_grid_hat.reshape(x1.shape)

svm4_grid_hat = svm4.predict(grid_show)
svm4_grid_hat = svm4_grid_hat.reshape(x1.shape)

svm5_grid_hat = svm5.predict(grid_show)
svm5_grid_hat = svm5_grid_hat.reshape(x1.shape)

svm6_grid_hat = svm6.predict(grid_show)
svm6_grid_hat = svm6_grid_hat.reshape(x1.shape)



cm_light = mpl.colors.ListedColormap(['#00ffcc', '#ffa0a0', '#a0a0ff'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.figure(facecolor='w', figsize=(14, 7))

##linnear-svm
plt.subplot(231)
##区域图
plt.pcolormesh(x1, x2, svm1_grid_hat, cmap=cm_light)
##所有样本点
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark) #样本
##测试数据集
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10) #圈中测试样本
##lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('鸢尾花数据SVM-c0.1分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

##svm
plt.subplot(232)
##区域图
plt.pcolormesh(x1, x2, svm2_grid_hat, cmap=cm_light)
##所有样本点
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark) #样本
##测试数据集
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10) #圈中测试样本
##lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('鸢尾花数据svm-c1分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

##poly-svm
plt.subplot(233)
##区域图
plt.pcolormesh(x1, x2, svm3_grid_hat, cmap=cm_light)
##所有样本点
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark) #样本
##测试数据集
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10) #圈中测试样本
##lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('鸢尾花数svm-c10分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

##sigmoid
plt.subplot(234)
##区域图
plt.pcolormesh(x1, x2, svm4_grid_hat, cmap=cm_light)
##所有样本点
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark) #样本
##测试数据集
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10) #圈中测试样本
##lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('鸢尾花数据svm-c100分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

##sigmoid
plt.subplot(235)
##区域图
plt.pcolormesh(x1, x2, svm5_grid_hat, cmap=cm_light)
##所有样本点
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark) #样本
##测试数据集
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10) #圈中测试样本
##lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('鸢尾花数据svm-c500分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

##sigmoid
plt.subplot(236)
##区域图
plt.pcolormesh(x1, x2, svm6_grid_hat, cmap=cm_light)
##所有样本点
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark) #样本
##测试数据集
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10) #圈中测试样本
##lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('鸢尾花数据svm-c10000分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

plt.suptitle(u'鸢尾花数据SVM分类器不同C参数效果比较', fontsize=16, color='r')
plt.show()