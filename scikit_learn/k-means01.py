#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '49、K-Means算法实战代码案例_笔记'
__author__ = 'Administrator'
__mtime__ = '2018\6\10 0010'
"""

import numpy as np
import  matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans

# 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

##产生模拟数据
#1、数据前期处理
n = 1500
centers = 4
x,y = ds.make_blobs(n, n_features=2, centers=centers, random_state=28)
x2,y2 = ds.make_blobs(n, n_features=2, centers=[(-10, -8), (-5,8), (5,2), (8,-7)], cluster_std=[1.5, 2.5, 1.9, 1], random_state=28)

#构造数据不平衡性 y==0时200条数据， y==1时100条
x3 = np.vstack((x[y==0][:200], x[y==1][:100], x[y==2][:10], x[y==3][:50]))
y3 = np.array([0]* 200 + [1]*100 + [2]*10 + [3]*50)
print(x3.shape)

#2\模型构建
km = KMeans(n_clusters=centers, init='random', random_state=28)
#n_clusters 就是k值，也就是聚类值
#init 初始化方，可以是kmeans++ 随机，或者 自定义ndarray
#y可要可不要，这里y的主要目的是为了让代码看效果一样，起点位符作用
km.fit(x, y)

y_hat = km.predict(x)
print('所有样本距离各自簇中心的总距离和:', km.inertia_)
print('平均距离：总距离/总样本数=',(km.inertia_ / n))
cluster_centers = km.cluster_centers_
print('k-means的各个族中心点坐标：',cluster_centers)
print('score其实就是所有样本点离所属簇中心点距离和的相反数=', km.score(x, y))

y_hat2 = km.fit_predict(x2)
y_hat3 = km.fit_predict(x3)

def expandBorder(a, b):
    d = (b - a) * 0.1
    return a - d, b + d

#3\画图
cm = mpl.colors.ListedColormap(list('rgbmyc'))
plt.figure(figsize=(15, 9), facecolor='w')

#子图1 2行4列
plt.subplot(241)
plt.scatter(x[:, 0], x[:,1], c=y, s=30, cmap=cm, edgecolors='none')

x1_min, x2_min = np.min(x, axis=0)
x1_max, x2_max = np.max(x, axis=0)
x1_min, x1_max = expandBorder(x1_min, x1_max)
x2_min, x2_max = expandBorder(x2_min, x2_max)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title('原始数据a')
plt.grid(True)

#子图2
plt.subplot(242)
plt.scatter(x[:, 0], x[:, 1], c=y_hat, s=30, cmap=cm, edgecolors='none')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title('k-means算法聚类结果b')
plt.grid(True)

m = np.array(((1, -5), (0.5, 5)))
#矩阵的点剩 数据旋转
data_r = x.dot(m)
y_r_hat = km.fit_predict(data_r)

#子图3
plt.subplot(243)
plt.scatter(data_r[:, 0], data_r[:, 1], c=y, s=30, cmap=cm, edgecolors='none')

x1_min,x2_min = np.min(data_r, axis=0)
x1_max,x2_max = np.max(data_r, axis=0)
x1_min, x1_max = expandBorder(x1_min, x1_max)
x2_min, x2_max = expandBorder(x2_min, x2_max)

plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title('数据旋转后原始数据图a')
plt.grid(True)

#子图4
plt.subplot(244)
plt.scatter(data_r[:, 0], data_r[:, 1], c=y_r_hat, s=30, cmap=cm, edgecolors='none')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title('数据旋转后k-means预测图b')
plt.grid(True)

#子图5
plt.subplot(245)
plt.scatter(x2[:, 0], x2[:, 1], c=y2, s=30, cmap=cm, edgecolors='none')
x1_min, x2_min = np.min(x2, axis=0)
x1_max, x2_max = np.max(x2, axis=0)
x1_min, x1_max = expandBorder(x1_min, x1_max)
x2_min, x2_max = expandBorder(x2_min, x2_max)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title('不同方差与中心点原始数据a')
plt.grid(True)

#子图6
plt.subplot(246)
plt.scatter(x2[:, 0], x2[:, 1], c=y_hat2, s=30, cmap=cm, edgecolors='none')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title('不同方差簇数据k-means算法预测结果b')
plt.grid(True)

#子图7
plt.subplot(247)
plt.scatter(x3[:, 0], x3[:, 1], c=y3, s=30, cmap=cm, edgecolors='none')
x1_min,x2_min = np.min(x3, axis=0)
x1_max,x2_max = np.max(x3, axis=0)
x1_min,x1_max = expandBorder(x1_min, x1_max)
x2_min,x2_max = expandBorder(x2_min, x2_max)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title('不同簇样本数据原始数据图a')
plt.grid(True)

#子图8
plt.subplot(248)
plt.scatter(x3[:, 0], x3[:, 1], c=y_hat3, s=30, cmap=cm, edgecolors='none')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
#set a title of the current axes
plt.title('不同簇样本数据k-means算法预测结果b')
plt.grid(True)

plt.tight_layout(2, rect=(0, 0, 1, 0.97))
plt.suptitle('a为原始数据b为预测数据-分布对k-means聚类影响',fontsize=16, color='r')
plt.show()

