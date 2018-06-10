#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '32\决策树优化策略：剪枝优化与随机森林_笔记'
__author__ = 'BfireLai'
__mtime__ = '2018/5/21'
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

rng = np.random.RandomState(10)
x = np.sort(10 *rng.rand(80, 1), axis=0)
y = np.sin(x).ravel()

# y[::5] 切片操作 从第一个开始每隔5个取一个数 当前例子中有16个数 16=80/5
y[::5] += 3 * (0.5 - rng.rand(16))

#构建不同深度的决策树
clf_0 = DecisionTreeRegressor(max_depth=1)
clf_1 = DecisionTreeRegressor(max_depth=2)
clf_2 = DecisionTreeRegressor(max_depth=3)
clf_3 = DecisionTreeRegressor(max_depth=5)
clf_0.fit(x, y)
clf_1.fit(x, y)
clf_2.fit(x, y)
clf_3.fit(x, y)

#创建预测模拟数据
x_test = np.arange(0.0, 10, 0.01).reshape(-1, 1)
print(x_test.shape)
y_0 = clf_0.predict(x_test)
y_1 = clf_1.predict(x_test)
y_2 = clf_2.predict(x_test)
y_3 = clf_3.predict(x_test)

#图表展示
plt.figure(figsize=(16,9), dpi=80, facecolor='w')

# 散点图
plt.scatter(x, y, c='k', s=30, label='train data')

#直线
plt.plot(x_test, y_0, c='y', label='max_depth=1,$r^2$=%.3f'%(clf_0.score(x, y)), linewidth=3)
plt.plot(x_test, y_1, c='g', label='max_depth=2,$r^2$=%.3f'%(clf_1.score(x, y)), linewidth=3)
plt.plot(x_test, y_2, c='r', label='max_depth=3,$r^2$=%.3f'%(clf_2.score(x, y)), linewidth=3)
plt.plot(x_test, y_3, c='b', label='max_depth=4,$r^2$=%.3f'%(clf_3.score(x, y)), linewidth=3)
plt.xlabel('x', horizontalalignment='left')
plt.ylabel('y=sin(x)')
plt.title('decision tree regression depth--overfitting')
plt.legend()
plt.show()
