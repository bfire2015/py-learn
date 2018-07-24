#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '68、SVM算法5个代码案例_加深理解SVM算法原理_笔记5'
__author__ = 'BfireLai'
__mtime__ = '2018/7/24'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import matplotlib.font_manager

from sklearn import svm #svm导入
from sklearn.exceptions import ChangedBehaviorWarning

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore', category=ChangedBehaviorWarning)

##模拟产生数据
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
##产生训练数据
x = 0.3 * np.random.randn(100, 2)
x_train = np.r_[x + 2, x - 2]
#产生测试数据
x = 0.3* np.random.randn(20, 2)
x_test = np.r_[x + 2, x -2]
#产生一些异常数据 20条
x_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

#模型训练
#错误率不超过1%
clf = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=0.1)
clf.fit(x_train)

#预测结果获取
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)
y_pred_outliers = clf.predict(x_outliers)
#返回1 表示属于这个类别，-1表示不属于这个类别
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
#离群点 应该是不属于一个类别
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

##获取绘图点的信息np.c_ 网格交叉
#Z的返回表示 ，越接近0就越属于那个类别
z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

print(z)
print(z.shape)


#画图
plt.figure(facecolor='w')
plt.title('异常点检测', fontsize=18, color='r')
#画出区域图
plt.contourf(xx, yy, z, levels=np.linspace(z.min(), 0, 9), cmap=plt.cm.PuBu)
#画圈圈
a = plt.contour(xx, yy, z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, z, levels=[0, z.max()], colors='palevioletred')

#画点
s = 40
b1 = plt.scatter(x_train[:, 0], x_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(x_test[:, 0], x_test[:, 1], c='blueviolet', s=s, edgecolors='k')
c = plt.scatter(x_outliers[:, 0], x_outliers[:, 1], c='gold', s=s, edgecolors='k')

##设置相关信息
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
		   ['分割超平面', '训练样本', '测试样本', '异常点'],
		   loc='upper left',
		   prop=matplotlib.font_manager.FontProperties(size=11)
		   )
plt.xlabel('训练集错误率%d/200;测试集错误率%d/40, 异常点错误率%d/40'%(n_error_train, n_error_test, n_error_outliers))
plt.show()

