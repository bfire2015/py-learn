#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '41、AdaBoost算法原理的举例推演_重点掌握_笔记'
__author__ = 'BfireLai'
__mtime__ = '2018/5/22'
"""

import numpy as np

def h(x):
	return np.sum([-p * np.log2(p) for p in x])

#print(h([1/2, 1/2]))

def alpha(e):
	return 0.5 * np.log2((1-e) / e)

print('阈值为2.5条件熵', 0.3 * h([1]) + 0.7 * h([3/7, 4/7]))
print('阈值为5.5条件熵', 0.6 * h([1/2, 1/2]) + 0.4 * h([3/4, 1/4]))
print('阈值为8.5条件熵', 0.9 * h([6/9, 3/9]) + 0.1 * h([1]))
print(alpha(0.3))

#加上了样本权限的情况
#a b c d
#a表示 y=-1 的概率，就是分割之后的子树
#b表示 y=1 的概率
#c表示 分割样本的左边概率
#d表示 分割样本的右边概率

c = 0.0582 * 3/1
d = 1-c
a = (4 * 0.0582)/(4 * 0.0582 + 3 * 0.1976)
b = 1-a
print('阈值为2.5条件熵', c * 0 + d * h([a, b]))

a = (0.0582 * 1) / (3 * 0.1976 + 1 * 0.0582)
b = 1-a
print('阈值为5.5条件熵', (6 * 0.0582) * h([1/2, 1/2]) + (1-6*0.0582) * h([a, b]))

a = (3 * 0.0582) / (3 * 0.1976 + 6 * 0.0582)
b = 1-a
print('阈值为8.5条件熵', (1- 0.0582) * h([a, b]) + 0.0582 * h([1]))

a1 = 0.6112
a2 = 1.1205
a3 = 1.631
print('对于第1条样本=', a1 + a2 - a3)
print('对于第3条样本=', a1 + a2 - a3)
print('对于第6条样本=', -a1 + a2 - a3)
print('对于第8条样本=', -a1 + a2 + a3)

