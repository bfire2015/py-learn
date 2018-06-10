#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '52、层次聚类思想与AGNES算法代码实战案例_笔记'
__author__ = 'Administrator'
__mtime__ = '2018\6\10 0010'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import warnings
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph ##knn的k近邻计算

# 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings(action='ignore', category=UserWarning)

#模拟数据生产 产生1000条
np.random.seed(10)
n_clusters = 4
n = 1000
#块状数据
data1, y1 = ds.make_blobs(n_samples=n, n_features=2, centers=((-1, 1), (1, 1), (1, -1), (-1, -1)), random_state=28)
#print(data1)
n_noise = int(0.1*n)
r = np.random.rand(n_noise, 2)
#print(r)
min1, min2 = np.min(data1, axis=0)
max1, max2 = np.max(data1, axis=0)
r[:, 0] = r[:, 0] * (max1-min1) + min1
r[:, 1] = r[:, 1] * (max2-min2) + min2

data1_noise = np.concatenate((data1, r), axis=0)
y1_noise = np.concatenate((y1, [4]*n_noise))
print(data1_noise.shape)

#拟合月牙形数据
data2,y2 = ds.make_moons(n_samples=n, noise=0.05)
data2 = np.array(data2)
n_noise = int(0.1 * n)
r = np.random.rand(n_noise, 2)
min1,min2 = np.min(data2, axis=0)
max1,max2 = np.max(data2, axis=0)
r[:, 0] = r[:, 0] * (max1 - min1) + min1
r[:, 1] = r[:, 1] * (max2 - min2) + min2
data2_noise = np.concatenate((data2, r), axis=0)
y2_noise = np.concatenate((y2, [3]*n_noise))
print(data2_noise.shape)

def expandBorder(a, b):
    d = (b - a) * 0.1
    return a-d, b+d

#画图
#给定画图的颜色
cm = mpl.colors.ListedColormap(list('rgbyc'))
plt.figure(figsize=(12, 10), facecolor='w')
linkages = ('ward', 'complete', 'average')

for index,(n_clusters, data, y) in enumerate(((4, data1, y1), (4, data1_noise, y1_noise),
                                               (2, data2, y2), (2, data2_noise, y2_noise))):
    # 前面的两个4表示几行几列，第三个参数表示第几个子图(从1开始，从左往右数)
    plt.subplot(4, 4, 4 * index + 1)
    plt.scatter(data[:, 0], data[:, 1], c=y, cmap=cm)
    plt.title('原始数据', fontsize=12)
    plt.grid(b=True, ls=':')
    min1,min2 = np.min(data, axis=0)
    max1,max2 = np.max(data, axis=0)
    plt.xlim(expandBorder(min1, max1))
    plt.ylim(expandBorder(min2, max2))

    # 计算类别与类别的距离(只计算最接近的七个样本的距离)
    # 希望在agens算法中，在计算过程中不需要重复性的计算点与点之间的距离
    # metric='minkowski', p=2 闵可夫斯基距离 p=2就是欧式距离
    connectivity = kneighbors_graph(data, n_neighbors=7, mode='distance', metric='minkowski', p=2, include_self=True)
    # 变成对称矩阵 方便查找距离的 减少一半的计算量
    connectivity = (connectivity + connectivity.T)
    # 根据簇间距离的度量方式
    for i,linkage in enumerate(linkages):
        ##进行建模，并传值
        print('簇的个数', n_clusters)
        #euclidean 奥几里得距离
        ac = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',connectivity=connectivity, linkage=linkage)
        #ac.fit(data)
        #y_hot = ac.labels_
        y_hot = ac.fit_predict(data)

        plt.subplot(4, 4, i + 2 + 4 * index)
        plt.scatter(data[:, 0], data[:, 1], c=y_hot, cmap=cm)
        plt.title(linkage, fontsize=14)
        plt.grid(b=True, ls=':')
        plt.xlim(expandBorder(min1, max1))
        plt.ylim(expandBorder(min2, max2))

plt.suptitle('agnes层次聚类的不同合并策略')
plt.tight_layout(0.5, rect=(0, 0, 1, 0.95))
plt.show()



