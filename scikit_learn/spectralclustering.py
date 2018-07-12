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
import warnings
from sklearn.cluster import spectral_clustering #引入谱聚类
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import euclidean_distances

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore', category=FutureWarning)

##创建模拟数据
n = 1000
centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]

#符合高斯分布的数据集
data1, y1 = ds.make_blobs(n, n_features=2, centers=centers, cluster_std=(0.75, 0.5, 0.3, 0.25), random_state=0)
data1= StandardScaler().fit_transform(data1)
#欧几里德
dist1 = euclidean_distances(data1, squared=True)
#权重计算公式 对象相似度的矩阵w x就是a参数 高斯相似度公式
affinity_params1 = map(lambda x:(x, np.exp(-dist1**2 / (x ** 2)) + 1e-6), np.logspace(-2, 0, 6))
print('sigma a参数：', np.logspace(-2, 0, 6))

#数据2
#圆形数据集
t=np.arange(0, 2*np.pi, 0.1)
data2_1 = np.vstack((np.cos(t), np.sin(t))).T
data2_2 = np.vstack((2 * np.cos(t), 2*np.sin(t))).T
data2_3 = np.vstack((3 * np.cos(t), 3 * np.sin(t))).T
data2 = np.vstack((data2_1, data2_2, data2_3))
y2 = np.concatenate(([0]* len(data2_1), [1]*len(data2_2), [2]*len(data2_3)))
##数据2的参数
dist2 = euclidean_distances(data2, squared=True)
affinity_params2 = map(lambda x:(x, np.exp(-dist2 **2 / (x ** 2)) + 1e-6), np.logspace(-2, 0, 6))

datasets = [(data1, y1, affinity_params1), (data2, y2, affinity_params2)]

def expandBorder(a, b):
	d = (b - a) * 0.1
	return a-d,b+d

colors = ['r', 'g', 'b', 'y']
cm = mpl.colors.ListedColormap(colors)

for i, (x, y, params) in enumerate(datasets):
	x1_min, x2_min = np.min(x, axis=0)
	x1_max, x2_max = np.max(x, axis=0)
	x1_min, x1_max= expandBorder(x1_min, x1_max)
	x2_min, x2_max = expandBorder(x2_min, x2_max)
	n_clusters = len(np.unique(y))
	plt.figure(figsize=(12, 8), facecolor='w')
	plt.suptitle('谱聚类-数据%d'%(i + 1), fontsize=20, color='r')
	plt.subplots_adjust(top=0.9, hspace=0.35)

	for j, param in enumerate(params):
		sigma, af = param
		#谱聚类的建模
		#af 指定相似度矩阵构造方式
		y_hat = spectral_clustering(af, n_clusters=n_clusters, assign_labels='kmeans', random_state=28)
		unique_y_hat = np.unique(y_hat)
		n_clusters = len(unique_y_hat) - (1 if -1 in y_hat else 0)
		print('第%d份数据的类别'%(i + 1), unique_y_hat, ";聚类簇数目:", n_clusters)

		##开始画图
		plt.subplot(3, 3, j + 1)
		for k, col in zip(unique_y_hat, colors):
			cur = (y_hat == k)
			plt.scatter(x[cur, 0], x[cur, 1], s=40, c=col, edgecolors='k')
		plt.xlim((x1_min, x1_max))
		plt.ylim((x2_min, x2_max))
		plt.grid(True)
		plt.title('$\sigma$=%.2f,聚类簇数目：%d'%(sigma, n_clusters), fontsize=16)
	plt.subplot(3, 3, 7)
	plt.scatter(x[:,0], x[:,1], c=y, s=30, cmap=cm, edgecolors='none')
	plt.xlim((x1_min, x1_max))
	plt.ylim((x2_min, x2_max))
	plt.title('原始数据，聚类簇数目:%d'%len(np.unique(y)))
	plt.grid(True)
	plt.show()