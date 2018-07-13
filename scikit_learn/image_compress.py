#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '图片压缩'
__author__ = 'BfireLai'
__mtime__ = '2018/7/12'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#图像的读写显库
from PIL import Image
from time import time
from scipy import misc

from sklearn.cluster import KMeans
#计算点与点之间最小距离，然后将点组成key/value键值对
from sklearn.datasets import load_sample_image
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

n_colors = 32

#读取图片，加载图片数据
image = Image.open('datas/ai111.png')
#将数据转换为rgb 的百分比，变成0-1之间的值
image = np.array(image, dtype=np.float64) / 255
#获取图片数据的维度、形状（长、宽、像素）
#彩色图片是三维 黑白是二维
original_shape= tuple(image.shape)
print('original_shape', original_shape)
shape_size = len(original_shape)
print('图片维度', shape_size)

#d设置为1是为了黑白图片的显示
w, h, d = original_shape[0], original_shape[1], 1
if shape_size == 3:
	d = original_shape[2]

#将图片数据设置为二维
image_v = np.reshape(image, (w * h, d))
#顺序打乱后取前面10000个点 相当于随机抽样
image_v_sample = shuffle(image_v, random_state=28)[:10000]
print(image_v_sample)

#算法构建
t0 = time()
k_means = KMeans(n_clusters=n_colors, random_state=28)
k_means.fit(image_v_sample)
print('模型构建-消耗时间：%.3fs'%(time() - t0))

#进行算法数据预测，得到图片对应所有的聚类类别信息
t0 = time()
labels = k_means.predict(image_v)
print('算法模型预测-消耗时间：%.3fs'% (time() - t0))

#进行随机数据抽取，得到图片对象的聚类的类别信息
t0 = time()
#随机排序 随机选择簇中心点
codebook_random = shuffle(image_v, random_state=28)[: n_colors]
labels_random = pairwise_distances_argmin(image_v, codebook_random)#计算两点最小距离
print('随机数据预测消耗时间：%.3fs'% (time() - t0))

"""
重新构建一个图片数据(压缩图片数据)
codebook：各个类别的具体像素值集合(聚类中心/簇中心/类别中心)
labels：原始图片各个像素点的类别集合
w: 原始/新图片宽度
h：原始/新图片高度
"""



def recreate_image(codebook, labels, w, h):
	"""Recreate the (compressed) image from the code book & labels"""
	d = codebook.shape[1]
	image = np.zeros((w, h, d))
	label_idx = 0 #第几个原始图像的像素点
	for i in range(w):
		for j in range(h):
			#获取原始数据像素点对应的类别中心点坐标
            # 再根据中心点坐标获取对应的像素值
			image[i][j] = codebook[labels[label_idx]]
			label_idx += 1
	return image

##图片显示
plt.figure('原始图片', facecolor='w')
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
if d == 1:
	plt.imshow(image, cmap=plt.cm.gray)
else:
	plt.imshow(image)

plt.figure('k-means算法压缩图片', facecolor='w')
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
km_image = recreate_image(k_means.cluster_centers_, labels, w, h)

km_image.shape = original_shape
if d == 1:
	plt.imshow(km_image, cmap=plt.cm.gray)
else:
	plt.imshow(km_image)

plt.figure('随机抽取像素点压缩图片', facecolor='w')
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
random_image = recreate_image(codebook_random, labels_random, w, h)
random_image.shape = original_shape
if d == 1:
	plt.imshow(random_image, cmap=plt.cm.gray)
else:
	plt.imshow(random_image)

plt.show()
misc.imsave('datas/result_1_original.png', image)
misc.imsave('datas/result_2_km.png', km_image)
misc.imsave('datas/result_3_random.png', random_image)