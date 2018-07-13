#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '57、聚类算法效果比较与应用案例：图片压缩_笔记'
__author__ = 'BfireLai'
__mtime__ = '2018/7/12'
"""

import time
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn import datasets as ds
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#产生模拟数据
n_samples= 1000
np.random.seed(0)

#产生圆形
noisy_circles = ds.make_circles(n_samples=n_samples, factor=.5, noise=.05)
#产生月牙形
noisy_moons = ds.make_moons(n_samples=n_samples, noise=.05)
#高斯分布h
blobs = ds.make_blobs(n_samples=n_samples, n_features=2, cluster_std=0.5, centers=3, random_state=0)
no_structure = np.random.rand(n_samples, 2), None

datasets = [noisy_circles, noisy_moons, blobs, no_structure]
clusters = [2, 2, 3, 2]

clustering_name = [
	'KMeans', 'MiniBatchKMeans', 'AC-ward', 'AC-average',
	'Birch', 'DBSCAN', 'SpectralClustering'
]

#开始画画
plt.figure(figsize=(len(clustering_name) * 2 + 3, 9.5), facecolor='w')
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96,wspace=.05, hspace=.01)
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)
plot_num = 1
for i_dataset, (dataset, n_cluster) in enumerate(zip(datasets, clusters)):
	x, y = dataset
	x = StandardScaler().fit_transform(x)
	connectivity = kneighbors_graph(x, n_neighbors=10, include_self=False)
	connectivity = 0.5 * (connectivity + connectivity.T)

	km = cluster.KMeans(n_clusters=n_cluster)
	mbkm = cluster.MiniBatchKMeans(n_clusters=n_cluster)
	ward = cluster.AgglomerativeClustering(n_clusters=n_cluster, connectivity=connectivity, linkage='ward')
	average = cluster.AgglomerativeClustering(n_clusters=n_cluster, connectivity=connectivity, linkage='average')
	birch = cluster.Birch(n_clusters=n_cluster)
	dbscan = cluster.DBSCAN(eps=.2)
	spectral = cluster.SpectralClustering(n_clusters=n_cluster, eigen_solver='arpack', affinity='nearest_neighbors')
	clustering_algorithms = [km, mbkm, ward, average, birch, dbscan, spectral]

	for name,algorithm in zip(clustering_name, clustering_algorithms):
		t0 = time.time()
		algorithm.fit(x)
		t1 = time.time()
		#如果模型中存在‘labels_’这个属性，那么获取这个预测的类型值
		if hasattr(algorithm, 'labels_'):
			y_pred = algorithm.labels_.astype(np.int)
		else:
			y_pred = algorithm.predict(x)

		#画子图
		plt.subplot(4, len(clustering_algorithms), plot_num)
		if i_dataset ==0:
			plt.title(name, size=18)
		plt.scatter(x[:, 0], x[:, 1], color=colors[y_pred].tolist(), s=10)
		#如果模型有中心点属性，那么画出中心点
		if hasattr(algorithm, 'cluster_centers)'):
			centers = algorithm.cluster_centers_
			center_colors = colors[:len(centers)]
			plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
		plt.xlim(-2, 2)
		plt.ylim(-2, 2)
		plt.xticks(())
		plt.yticks(())
		plt.text(.99, .01, ('%2fs'%(t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=15, horizontalalignment='right')
		plot_num += 1

#保存图片
plt.savefig('datas/ai111.png', dpi=200)
plt.show()