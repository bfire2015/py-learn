#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '59、有约束的最优化问题：拉格朗日乘子法、KKT条件_笔记'
__author__ = 'BfireLai'
__mtime__ = '2018/7/12'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#拉格郞日乘子法理解
def f(x, y):
	return 0.6 * (x + y) ** 2 - x*y

#构造数据
x1 = np.arange(-8, 8, 0.2)
x2 = np.arange(-8, 8, 0.2)
x1, x2 = np.meshgrid(x1, x2)
y = np.array(list(map(lambda t: f(t[0], t[1]), zip(x1.flatten(), x2.flatten()))))
y.shape = x1.shape

#限制条件
x3 = np.arange(-4, 4, 0.2)
x3.shape = 1,-1
x4 = np.array(list(map(lambda t:t**2 -t + 1, x3)))

#画图
fig = plt.figure(figsize=(12, 8), facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap=plt.cm.jet)
ax.plot(x3, x4, 'ro--', linewidth=2)

ax.set_title('函数y=0.6*(q1 +q2)^2-q1*q2')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#plt.savefig('datas/lagrange.png', dpi=200)
plt.show()
