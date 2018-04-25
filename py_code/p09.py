#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2017/12/25'
"""

#09、对ndarray的操作：索引、切片、转置
# 多维数组的索引 取元素的值
# axis0 axis1
# 索引--->数组的切片
# 在各个维度上单独切片，如果某维度都保留，则直接使用:冒号
import numpy as np

#a = np.random.random((2, 3, 4))
#print(a)
#print(a[0][1][2])
#print(a[0, 1, 2])

#print(a[0][0][1:3])
#print(a[0][1][1:3])
#print(a[0][2][1:3])
#print(a[0, :, 1:3])
#print('-------')
#print(a[0][:].T[1:3].T)

#-------------------------------#
# 注意：numpy中通过切片得到的新数组，只是原来数组的一个视图，因此对新数组
# 进行操作也会影响原数组  内存地址的引用还是一样
# 布尔值的索引：利用布尔类型的数组进行数据索引，最终返回的结果是对应
# 索引数组中数据为True位置的值
# True位置的元素取出来形成一个新数组(一维的)
# 条件是：b与c数组的shape必须一致
# 数据提取 数据清洗用到的方法
# 注意：numpy中不能使用python中的and or not操作符
# 使用 & | ~来替换
#print('bool值的索引')
#b = np.random.random((4, 4, 2))
#print(b)
#c = b > 0.5
#print(c)
#d = b[c]
#print(d)



# 花式索引：利用整数数组进行索引的方式
# ix_函数会产生一个索引器
e = np.arange(32).reshape((8, -1))
print(e)
print('---------')
print('连续:', e[1:3])
print('非连续行,连续的列:', e[[0, 3, 5]])
print('非连续行 非连续列  交叉的值:', e[[0, 3, 5], [0, 3, 2]])
print('非连续行 非连续列 所有的', e[np.ix_([0, 3, 5], [0, 2 ,3])])
print('---------')
print(e[[0, 3, 5]].T[[0, 2 ,3]].T)

# 数组转置与轴对换
# 数组转置是指将shape进行重置操作，并将其值重置为原始shape元祖的倒置
# 比如原始shape为（2,3,4） 则转置后为（4,3,2）
# 对于二维数组矩阵而言，数组的转置就是矩阵的转置
# 可以通过调用数组的transpose函数或T属性进行数组转置操作

#shape()
print('shape--------')
f = np.arange(24).reshape((2, 3, 4))
print(f)
g = f.T
g = np.transpose(f)
g = f.transpose()
print(g, g.shape)

