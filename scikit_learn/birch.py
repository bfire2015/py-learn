#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '53、层次聚类算法优化：BIRCH算法及其实战代码案例_笔记'
__author__ = 'Administrator'
__mtime__ = '2018\6\10 0010'
"""

from itertools import cycle
from time import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.datasets.samples_generator import make_blobs

# 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#产生模拟数据
xx = np.linspace(-22, 22, 10)
yy = np.linspace(-22, 22, 10)
xx, yy = np.meshgrid(xx, yy)
#10*10 = 100个中心点网格交叉形成
n_centres = np.hstack((np.ravel(xx)[:, np.newaxis], np.ravel(yy)[:, np.newaxis]))

#产生10万条特征属性是2，类别是100,符合高斯分布的数据集
x,y = make_blobs(n_samples=10000, n_features=2, centers=n_centres, random_state=28)

##创建不同的参数（簇直径）Birch层次聚类
birch_models = [
    Birch(threshold=1.7, n_clusters=None),
    Birch(threshold=0.5, n_clusters=None),
    Birch(threshold=1.7, n_clusters=100)
]
#threshold：簇直径的阈值，    branching_factor：分支因子

# 课后扩展：我们也可以加参数来试一下效果，比如加入分支因子branching_factor，给定不同的参数值，看聚类的结果

## 画图
final_step = ['类直径=1.7;n_lusters=none', '类直径=0.5;n_lusters=none', '类直径=1.7,n_lusters=100']
plt.figure(figsize=(12, 8), facecolor='w')
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.9)
colors_ = cycle(colors.cnames.keys())
cm = mpl.colors.ListedColormap(colors.cnames.keys())

for index, (birch_model, info) in enumerate(zip(birch_models, final_step)):
    t = time()
    birch_model.fit(x)
    time_ = time() - t
    #获取模型结果（labels_ 和中心点）
    labels = birch_model.labels_
    centroids = birch_model.subcluster_centers_
    n_clusters = len(np.unique(centroids))
    print('子类别中心数目%d'%n_clusters)
    print('birch算法，参数信息为%s,模型构建消耗时间：%.3f秒，聚类中心数目：%d'%(info, time_, len(np.unique(labels))))

    ##画图
    sub_index = 221 + index
    plt.subplot(sub_index)

    for this_centroid, k, col in zip(centroids, range(n_clusters), colors_):
        mask = labels == k
        plt.plot(x[mask, 0], x[mask, 1], 'w', markerfacecolor=col, marker='.')
        if birch_model.n_clusters is None:
            plt.plot(this_centroid[0], this_centroid[1], '*', markerfacecolor=col, markeredgecolor='k', markersize=2)
    plt.ylim([-25, 25])
    plt.xlim([-25, 25])
    plt.title('birch算法%s,耗时%.3fs'%(info, time_))
    plt.grid(False)

##原始数据集显示
plt.subplot(224)
plt.scatter(x[:, 0], x[:, 1], c=y, s=1, cmap=cm, edgecolors='none')
plt.ylim([-25, 25])
plt.xlim([-25, 25])
plt.title('原始数据')
plt.grid(False)

plt.suptitle('不同参数下birch算法比较')
plt.tight_layout(0.5, rect=(0, 0, 1, 0.95))
plt.show()