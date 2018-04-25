#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/1/11'
"""
#print(mpl.__version__)
# python最常用的绘图库，提供了一整套十分适合交互式绘图的命令API，比较方便的就可以将其嵌入到
# GUI应用程序中
# 官网：http://matplotlib.org/

# figure和subplot
# Figure：面板图，matplotlib中的所有图像都是位于figure对象中，一个图像只能有一个figure对象 类似于一块黑板 画板
# Subplot:子图，figure对象下创建一个或多个subplot对象（即axes）用于绘制图表

# 设置figsize=8*6 分辨率为80
# defaults to rc figure.figsize
# 获取figure对象，方便在其上面创建子图即axes对象
# 获取所有的自带样式 # 设置图形的显示风格
# 颜色 标记 线型 color marker linestyle
# 设定x轴 y轴的范围 刻度 标签
# 保存图片文件 必须放到show方法之前
# plt.savefig('ai111.png', dpi=200)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
print(mpl.__version__)
# defaults to rc figure.figsize
print(mpl.rcParams['figure.figsize'])
# 取到figure对象 方便我们创建子图 axes对象
figure = plt.figure(figsize=(6, 8), dpi=80)
axes1 = figure.add_subplot(2, 2, 1)
# axes2 = figure.add_subplot(2, 2, 2)
# axes3 = figure.add_subplot(2, 2, 3)
# axes4 = figure.add_subplot(2, 2, 4)

# 设置图标的显示风格
print(plt.style.available)
# plt.style.use('ggplot')

x = np.arange(-5, 5)
# x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
y = np.sin(x)
print(x, y)
# 设置x轴 y轴的范围
# plt.axis([-5, 5, -5, 5])
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel(u'x轴')
plt.ylabel(u'y轴')
plt.xticks(np.arange(-3, 3))
plt.yticks(np.arange(-1, 3))
plt.title(u'标题')
plt.text(-3, -3, 'y=sin(x)', fontsize=20, bbox={'facecolor': 'yellow', 'alpha': 0.2})
# plt.grid = True
# 颜色 线型 标记
# help(plt.plot)
plt.plot(x, y, color='r', linestyle='--', marker='o')

# 保存图片
plt.savefig('ai111.png', dpi=200)
# 去除右边和上边的边框
axes1.spines['right'].set_color('None')
axes1.spines['top'].set_color('None')


# plt.legend()
plt.show()

