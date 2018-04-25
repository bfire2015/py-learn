#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/1/11'
"""

# pandas:层次索引
# 在某一个方向拥有多个（两个及两个以上）索引级别
# 通过层次化索引，pandas能够以较低维度形式处理高纬度的数据
# 通过层次化索引，可以按照层次统计数据
# 层次索引包括Series层次索引和DataFrame层次索引

#s1 = pd.Series(data=[99, 87, 76, 67, 99],index=[['2017', '2017', '2018', '2018', '2018'], ['dfy', 'lisi', 'dfy', 'lisi', 'zs']])
# 数据透视表 用到层次索引

# df1 = pd.DataFrame({
#     'year': [2016, 2016, 2017, 2017, 2018],
#     'fruit': ['apple', 'banana', 'apple', 'banana', 'apple'],
#     'production': [2345, 3242, 5667, 2576, 2134],
#     'profits': [23.22, 76.89, 90.99, 78.22, 98.76],
# })

# ix是比较老的方法 新方式是使用iloc  loc
# iloc 对下标值进行操作 Series与DataFrame都可以操作
# loc 对索引值进行操作 Series与DataFrame都可以操作

# 将数据进行归一 合并  数据有重复  类似合并单元格
# 设置多个行索引 层次化索引
# 根据层次化索引取值 此时可以理解为三维的

# pandas：按照层次索引进行统计数据
# print(df2.sum(level='year'))
# print(df2.mean(level='fruit'))
# 下面的level写了2个相当于是 并且的关系
# print(df2.min(level=['year', 'fruit']))

import pandas as pd
import numpy as np

print(pd.__version__)
s1 = pd.Series(data=[99, 87, 76, 67, 99],index=[['2017', '2017', '2018', '2018', '2018'], ['dfy', 'lisi', 'dfy', 'lisi', 'zs']])
# 数据透视表 用到层次索引
print(s1)
print(s1.iloc[1])
print(s1.loc['2017', 'lisi'])

print('-----')
df1 = pd.DataFrame({
    'year': [2016, 2016, 2017, 2017, 2018],
    'fruit': ['apple', 'banana', 'apple', 'banana', 'apple'],
    'production': [2345, 3242, 5667, 2576, 2134],
    'profits': [23.22, 76.89, 90.99, 78.22, 98.76],
})
print(df1)
print('-----')
df2 = df1.set_index(['year', 'fruit'])
# df1.index = ['a', 'b', 'a', 'c', 'd']
# print(df2)
# print(df2.index)
print('------')
# print(df2.sum(level='year'))
# print(df2.mean(level='fruit'))
# print(df2.sum(level=['year', 'fruit']))

#取值的新方法
print(df1['profits'][2])
print(df1.ix[2]['profits'])
#iloc 对下标值进行操作
print(df1.iloc[2, 2])
#loc 对索引值进行操作
print(df1.loc[2, 'profits'])
