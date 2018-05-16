#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '梯度下降法原理与代码实战案例_笔记'
__author__ = 'BfireLai'
__mtime__ = '2018/5/15'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#构建一维元素图像
def f1(x):
	return  0.5 * (x - 0.25) ** 2

def h1(x):
	return  0.5 * 2 * (x - 0.25)

#使用梯度下降法进行解答

GD_X = []
GD_Y = []
x = 4
#alpha = 0.5
alpha = 1.2
f_change = f1(x)
f_current = f_change

#迭代次数
iter_num = 0

while f_change > 1e-10 and iter_num < 100:
	iter_num += 1
	x = x - alpha * h1(x)
	tmp = f1(x)
	#判断y值 的变化 不能太小
	f_change = np.abs(f_current - tmp)
	f_current = tmp
	GD_X.append(x)
	GD_Y.append(f_current)

print('最终的结果：（%.5f,%.5f）'%(x, f_current))
print('迭代次数是：%d'%iter_num)
print(GD_X)

#构建数据
X = np.arange(-4, 4.5, 0.05)
Y = np.array(list(map(lambda t: f1(x), X)))

#画图
plt.figure(facecolor='w')
plt.plot(X, Y, 'r-', linewidth=2)
plt.plot(GD_X, GD_Y, 'bo--', linewidth=2)
plt.title('函数$y = 0.5*(x-0.25)^2$;学习率:%.3f;最终解:（%.3f,%.3f）;迭代次数:%d'%(alpha, x, f_current, iter_num))
plt.show()


