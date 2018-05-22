#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '42、AdaBoost算法的实战代码案例与总结_笔记'
__author__ = 'BfireLai'
__mtime__ = '2018/5/22'
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.ensemble import AdaBoostClassifier #adaboost引入方法
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles #造数据

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#创建数据
#创建符合高斯分布的数据集 cov方差，std标准差,mean均值，默认值为0
x1,y1 = make_gaussian_quantiles(cov=2, n_samples=200, n_features=2, n_classes=2, random_state=1)
x2,y2 = make_gaussian_quantiles(mean=(3,3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=1)
print(x1.shape, y1.shape)
print(x2.shape, y2.shape)

x = np.concatenate((x1, x2))
y = np.concatenate((y1, -y2 +1))

print(x.shape, y.shape)

#构建adaboost模型
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm='SAMME.R', n_estimators=200)
#数据量大的时候，可以增加内部分类器的树深度，也可以不限制树深
#max_depth树深，数据量大的时候，一般范围在10——100之间
#数据量小的时候，一般可以设置树深度较小，或者n_estimators较小

#n_estimators 迭代次数或者最大弱分类器数：200次
#base_estimator：DecisionTreeClassifier 选择弱分类器，默认为CART树
#algorithm(运算方式)：SAMME 和SAMME.R 。运算规则，后者加.R的是Real算法，以概率调整权重（），迭代速度快（更快找到最优的），
#需要能计算概率的分类器支持
#learning_rate：0<v<=1，默认为1，正则项 衰减指数 默认是1可以调小 就是在更新样本权重的时候不要变化得太快
#loss：linear、‘square’ 、exponential’。误差计算公式：一般用linear足够 回归问题才有 误差率的表示方式

bdt.fit(x, y)

#画图
plot_step = 0.02
x1_min,x1_max = x[:,0].min() - 1, x[:,0].max() +1
x2_min,x2_max = x[:,1].min() - 1, x[:,1].max() +1
#网格
xx,yy = np.meshgrid(np.arange(x1_min, x1_max, plot_step), np.arange(x2_min, x2_max, plot_step))

#预测
z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
#设置维度
z = z.reshape(xx.shape)

plot_colors = 'br'
class_names = 'AB'

plt.figure(figsize=(10,5), facecolor='w')
#局部子图
#121 表示在1行2列里面 最后 一个1表示在第一个位置，第一个子图
plt.subplot(121)
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Paired)

for i, n, c in zip(range(2), class_names, plot_colors):
	idx = np.where(y == i)
	#散点图
	plt.scatter(x[idx, 0], x[idx, 1], c=c, cmap=plt.cm.Paired, label='类别%s'%n)


plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.legend(loc='upper right')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Adaboost算法分类，正确率为%.2f%%'%(bdt.score(x, y) * 100))

#获取决策函数的数值
#决策函数 最终的200弱学习器整合之后的，计算所有样本预测值
twoclass_output = bdt.decision_function(x)
print('获取决策函数的数值=', twoclass_output)
print('获取决策函数的数值（长度）=', len(twoclass_output))

#获取范围
plot_range = (twoclass_output.min(), twoclass_output.max())

#122 表示在1行2列里面 最后一个2表示在第二个位置，第二个子图
plt.subplot(122)

for i,n,c in zip(range(2), class_names, plot_colors):
	#直方图
	plt.hist(twoclass_output[y == i], bins=20, range=plot_range, facecolor=c, label='类别%s'%n, alpha=.5)

x1,x2,y1,y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('样本数')
plt.xlabel('决策函数值')
plt.title('adaboost的决策值')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()

