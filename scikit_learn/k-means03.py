#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '51、聚类算法的衡量指标_轮廓系数_笔记'
__author__ = 'Administrator'
__mtime__ = '2018\6\10 0010'
"""

import time
import matplotlib as mpl
from sklearn.cluster import MiniBatchKMeans,KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

# 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

centers = [[1, 1], [-1, -1], [1, -1]]
clusters = len(centers)

x,y = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7, random_state=28)
#y 值在实际工作中是人工给定的， 专门用于判断聚类的效果的一个值
# 实际工作中，我们假定聚类算法的模型都是比较可以的，最多用轮廓系数或模型的api score返回值进行度量；
# 其它的效果度量方式一般不用 原因是：其它度量方式需要给定数据的实际的y值 那么当我们知道y值的时候，其实可以直接使用分类算法，不需要使用聚类

k_means = KMeans(init='k-means++', n_clusters=clusters, random_state=28)
t0 = time.time()
k_means.fit(x)
km_batch = time.time() - t0
print('k-means算法模型训练消耗时间：%.4fs'% km_batch)

batch_size = 100
mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, batch_size=batch_size, random_state= 28)
t0 = time.time()
mbk.fit(x)
mbk_batch = time.time() - t0
print('mini batch k-means 算法模型训练消耗时间：%.4fs'%mbk_batch)
print('k-means算法模型-所有样本点离所属簇中心点距离和的相反数据：', k_means.score(x))
print('mini batch k-means-score值：', mbk.score(x))

km_y_hat = k_means.labels_
mbkm_y_hat = mbk.labels_
print(km_y_hat)#样本所属的类别

k_means_cluster_centers = k_means.cluster_centers_
mbk_means_cluster_centers = mbk.cluster_centers_
print('k-means算法聚类中心点：\ncenters=', k_means_cluster_centers)
print('mini bathc k-means 算法聚类中心点：\ncenters=', mbk_means_cluster_centers)
order = pairwise_distances_argmin(k_means_cluster_centers, mbk_means_cluster_centers)

#方便后面比较预测的不同样本
print('对应的顺序：', order)

###效果评估
score_funcs = [
    metrics.adjusted_rand_score, #api
    metrics.v_measure_score,#均一性和完整性的加权平均
    metrics.adjusted_mutual_info_score, #ami 调整互信息
    metrics.mutual_info_score, #互信息
]

##2、迭代对每个评估函数进行评估操作
for score_func in score_funcs:
    t0 = time.time()
    km_scores = score_func(y, km_y_hat)
    print('k-means算法:%s评估函数计算结果值：%.5f;指标计算消耗时间：%0.3fs'%(score_func.__name__, km_scores, time.time() - t0))
    t0 = time.time()
    mbk_scores = score_func(y, mbkm_y_hat)
    print('mini batch k-means算法：%s评估函数计算结果值：%.5f;指标计算消耗时间：%0.3fs\n'%(score_func.__name__, mbk_scores, time.time() - t0))



print('all the end')


