#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/7/11'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

##创建模拟数据
N = 1000
centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
data1, y1 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(1, 0.75, 0.5, 0.25), random_state=0)
data1 = StandardScaler().fit_transform(data1)

params1 = ((0.2, 5), (0.2, 10), (0.2, 15), (0.3, 5), (0.3, 10), (0.3, 15))

t = np.arange(0, 2*np.pi, 0.1)
data2_1 = np.vstack((np.cos(t), np.sin(t))).T
data2_2 = np.vstack((2*np.cos(t), 2*np.sin(t))).T
data2_3 = np.vstack((3*np.cos(t), 3*np.sin(t))).T
data2 = np.vstack((data2_1, data2_2, data2_3))
y2 = np.concatenate(([0] * len(data2_1), [1] * len(data2_2), [2] * len(data2_3)))
params2 = ((0.5, 3), (0.5, 5), (0.5, 10), (1., 3), (1., 10), (1., 20))

datasets = [(data1, y1, params1), (data2, y2, params2)]

def expandBorder(a, b):
	d = (b - a) * 0.1
	return a-d, b+d

colors = ['r', 'g', 'b', 'y', 'c', 'k']
cm = mpl.colors.ListedColormap(colors)

for i,(x, y, params) in enumerate(datasets):
	x1_min, x2_min = np.min(x, axis=0)
	x1_max, x2_max = np.max(x, axis=0)
	x1_min, x1_max = expandBorder(x1_min, x1_max)
	x2_min, x2_max = expandBorder(x2_min, x2_max)

	plt.figure(figsize=(12,8), facecolor='w')
	plt.suptitle('dbscan密度聚类-数据%d '%(i + 1), fontsize=20, color='r')
	plt.subplots_adjust(top=0.9, hspace=0.35)

	for j,params in enumerate(params):
		eps, min_samples = params
		model = DBSCAN(eps=eps, min_samples=min_samples)
		#eps 半径，控制领域的大小，值越大，越能容忍噪声点，值越小，相比形成的簇就越多
		#min_samples 原理中所说的m, 控制哪个是核心点，值越小，越可容忍噪声点，越大，就更容易把有效点划分成噪点

		model.fit(x)
		y_hat = model.labels_

		unique_y_hat = np.unique(y_hat)
		#-1 在预测值里表示不知道该样本属于哪个簇，需要减去一个簇类别
		n_clusters = len(unique_y_hat) - (1 if -1 in y_hat else 0)
		print("第%d份数据：预测的所有类别"%(i + 1), unique_y_hat, ";聚类簇数目:", n_clusters)

		core_samples_mask = np.zeros_like(y_hat, dtype=bool)
		#标注核心样本点core_sample_indices_
		core_samples_mask[model.core_sample_indices_] = True

		##开始画图
		plt.subplot(3, 3, j + 1)
		for k, col in zip(unique_y_hat, colors):
			if k == -1:
				col = 'k'

			class_member_mask = (y_hat == k)
			#核心样本点
			xy = x[class_member_mask & core_samples_mask]
			plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
			xy = x[class_member_mask & ~core_samples_mask]
			plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

		plt.xlim((x1_min, x1_max))
		plt.ylim((x2_min, x2_max))
		plt.grid(True)
		plt.title('$\epsilon$=%.1f m=%d, 聚类簇数目：%d'%(eps, min_samples, n_clusters), fontsize=14)
	##原始数据显示
	plt.subplot(3, 3, 7)
	plt.scatter(x[:, 0], x[:, 1], c=y,s=30, cmap=cm, edgecolors='none')
	plt.xlim((x1_min, x1_max))
	plt.ylim((x2_min, x2_max))
	plt.title('原始数据集，聚类簇数目%d'%len(np.unique(y)))
	plt.grid(True)
	plt.show()