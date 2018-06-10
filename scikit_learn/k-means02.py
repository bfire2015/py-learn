#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '50、Mini Batch K-Means算法原理及其实战代码案例_笔记'
__author__ = 'Administrator'
__mtime__ = '2018\6\10 0010'
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import MiniBatchKMeans,KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import  pairwise_distances_argmin

# 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#初始化三个中心
centers = [[1, 1], [-1, -1], [1, -1]]
clusters = len(centers)
x, y = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7, random_state=28)

#构建kmeans算法
k_means = KMeans(init='k-means++', n_clusters=clusters, random_state=28)
t0 = time.time() #当前时间
k_means.fit(x)#训练模型
km_bathch = time.time() - t0 #使用kmeans训练数据消耗时间
print('k-means算法模型训练消耗时间%.4fs'%km_bathch)

#构建MinBatchKMeans 算法
batch_size = 100
mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, batch_size=batch_size, random_state=28)
t0 = time.time()
mbk.fit(x)
mbk_batch = time.time()-t0
print('mini batch k-means 算法模型训练消耗时间%.4fs'%mbk_batch)

#预测结果
km_y_hat = k_means.predict(x)
mbkm_y_hat = mbk.predict(x)

print(km_y_hat[:10])
print(mbkm_y_hat[:10])
print(k_means.cluster_centers_)
print(mbk.cluster_centers_)

#获取聚类中心点并与聚类中心点进行排序
k_means_cluster_centers = k_means.cluster_centers_ #输出k-means聚类中心点
mbk_means_cluster_centers = mbk.cluster_centers_ #输出mbk聚类中心点
print('k-means算法聚类中心：\n center=', k_means_cluster_centers)
print('mini batch k-means算法聚类中心点:\ncenter=', mbk_means_cluster_centers)
order = pairwise_distances_argmin(k_means_cluster_centers, mbk_means_cluster_centers)
#方便后面比较预测的不同样本
print('对应的顺序：', order)

#画图
plt.figure(figsize=(12, 6), facecolor='w')
#子图的上下左右边距
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
#设置样本点的颜色
cm = mpl.colors.ListedColormap(['#ffc2cc', '#c2ffcc', '#ccc2ff'])
#设置簇中心点颜色
cm2 = mpl.colors.ListedColormap(['#ff0000', '#00ff00', '#0000ff'])
#子图1：原始数据
plt.subplot(221)
plt.scatter(x[:, 0], x[:, 1], c=y, s=6, cmap=cm, edgecolors='none')
plt.title('原始数据分布图')
plt.xticks(())
plt.yticks(())
plt.grid(True)

#子图2 k-means算法聚类结果图
plt.subplot(222)
plt.scatter(x[:, 0], x[:, 1], c=km_y_hat, s=6, cmap=cm, edgecolors='none')
plt.scatter(k_means_cluster_centers[:, 0], k_means_cluster_centers[:, 1], c=range(clusters), s=60, cmap=cm2, edgecolors='none')
plt.title('k-means算法聚类预测图')
plt.xticks(())
plt.yticks(())
plt.text(-3.8, 3, '模型训练耗时：%.2fms'%(km_bathch * 1000))
plt.grid(True)

#子图3 mini batch k-means算法聚类结果图
plt.subplot(223)
plt.scatter(x[:, 0], x[:, 1], c=mbkm_y_hat, s=6, cmap=cm, edgecolors='none')
plt.scatter(mbk_means_cluster_centers[:, 0], mbk_means_cluster_centers[: ,1], c=range(clusters), s=60, cmap=cm2, edgecolors='none')
plt.title('mini bathc k-means算法聚类结果图')
plt.xticks(())
plt.yticks(())
plt.text(-3.8, 3, '模型训练耗时：%.2fms'%(mbk_batch*1000))
plt.grid(True)

#mini batch k-means 预测由边界点 -1
different = list(map(lambda a:(a!= 0) & (a!=1) & (a!=2), mbkm_y_hat))
#中0 1 2 3
for k in range(clusters):
    different += (km_y_hat == k) != (mbkm_y_hat == order[k])

#逻辑取反
identic = np.logical_not(different)
#计算一共多少个不同的样本点
different_nodes = len(list(filter(lambda b:b, different)))

#子图4 标出mini batch k-means and k-means 预测不同的样本点
plt.subplot(224)
#两者预测相同的
plt.plot(x[identic, 0], x[identic, 1], 'w', markerfacecolor='#bbbbbb', marker='.')
#两者预测不相同的
plt.plot(x[different, 0], x[different, 1], 'w', markerfacecolor='m', marker='.', markersize=12)
plt.title('mini batch k-means and k-means算法预测结果不同的点')
plt.xticks(())
plt.yticks(())
plt.text(-3.8, 2, '预测不同点：%d个'%(different_nodes))
plt.grid(True)

plt.suptitle('mini bathc k-means and k-means 比较')
plt.show()





