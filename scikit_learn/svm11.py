#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '69、SVM算法4个实战综合案例_笔记1'
__author__ = 'BfireLai'
__mtime__ = '2018/7/24'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm,datasets

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

##数据加载
iris = datasets.load_iris()
##获取前两列作为特征属性
x = iris.data[:, :2]
y = iris.target

##自定义一个核函数，参数xy是特征性矩阵
##xy就是样本向量
def my_kernel(x, y):
	"""
	we create a custom kernel:(2 0)
	k(x,y) = x()y.T
	(0 1)
	:param x:
	:param y:
	:return:
	"""
	m = np.array([[2, 0], [0, 1]])
	return np.dot(np.dot(x, m), y.T)

##使用自定义的核函数创建一个svm对象
clf = svm.SVC(kernel=my_kernel)
clf.fit(x, y)

##评估效果
score = clf.score(x, y)
print('训练集准确率%.2f%%'%(score*100))

##构建预测网格
h = .02
x_min, x_max = x[:, 0].min() - 1, x[:,0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:,1].max() + 1
xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
##预测值
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.figure(facecolor='w', figsize=(8, 4))

##画出区域图
plt.pcolormesh(xx, yy, z, cmap=cm_light)

##画出点图
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_dark)

##设置title及其它相关属性
plt.title('svm中自定义核函数', color='r', fontsize=16)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.text(x_max - .3, y_min + .3,('准确率%.2f%%'%(score * 100)).lstrip('0'), size=15, horizontalalignment='right')
plt.show()