#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '梯度下降法原理与代码实战案例 - 二维原始图像2'
__author__ = 'BfireLai'
__mtime__ = '2018/5/15'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#设置二维原始图像
def f2(x, y):
	return 0.15 * (x + 0.5) **2 + 0.25 * (y - 0.25)**2 + 0.35 * (1.5*x - 0.2*y + 0.35)**2

##偏导数
def hx2(x, y):
	return 0.15 * 2 * (x+ 0.5) + 0.25*2*(1.5*x - 0.2*y + 0.35) * 1.5

def hy2(x, y):
	return 0.25*2*(y - 0.25)-0.25*2*(1.5*x - 0.2*y + 0.35) *0.2

#使用梯度下降法求解
GD_X1 = []
GD_X2 = []
GD_Y = []
x1 = x2 =4
alpha = 0.5
f_change = f2(x1, x2)
f_current = f_change
GD_X1.append(x1)
GD_X2.append(x2)
GD_Y.append(f_current)
iter_num = 0
while f_change > 1e-10 and iter_num < 100:
	iter_num += 1
	prex1 = x1
	prex2 = x2
	x1 = x1 - alpha * hx2(prex1, prex2)
	x2 = x2 - alpha * hy2(prex1, prex2)

	tmp = f2(x1, x2)
	f_change = np.abs(f_current - tmp)
	f_current = tmp
	GD_X1.append(x1)
	GD_X2.append(x2)
	GD_Y.append(f_current)

print('最终结果为:(%.5f,%.5f,%.5f)'%(x1, x2, f_current))
print('迭代过程中x的取值，迭代次数:%d'%iter_num)
print(GD_X1)

#构建数据
X1 = np.arange(-4, 4.5, 0.2)
X2 = np.arange(-4, 4.5, 0.2)
X1,X2 = np.meshgrid(X1, X2)
Y = np.array(list(map(lambda t:f2(t[0], t[1]), zip(X1.flatten(), X2.flatten()))))
Y.shape = X1.shape

#画图
fig = plt.figure(facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=plt.cm.jet)
ax.plot(GD_X1, GD_X2, GD_Y, 'ko--')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_title('函数\n学习率:%.3f,最终解:(%.3f,%.3f,%.3f),迭代次数:%d'%(alpha, x1, x2, f_current, iter_num))
plt.show()

