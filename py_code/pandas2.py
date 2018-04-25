#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/1/10'
"""
# Series值的获取的两种方式：
# 1、通过方括号+索引的方式获取对应索引的数据，可能返回多条数据
# 2、通过方括号+下标值的方式获取数据，下标值的取值范围为：[0, len(Series.values))；另外下标值也可以是负数，表示从右往左获取数据

# Series获取多个值的方式类似Numpy中的ndarray的切片操作
# 通过方括号+下标值/索引值+冒号(的形式来截取series对象中的一部分数据
import pandas as pd
import numpy as np

dict1 = {'语文':77, '数学':88, '英语':99}
s1 = pd.Series(dict1, dtype=np.float)
# print(s1)
# print(s1['英语':'语文'])
# print(s1[1:2])

arr1 = np.array([100, 20, 30])
# print(s1 + arr1)

s2 = pd.Series({'bf': 100, 'zs': 88})
print(s2)
# s2.index = ['a', 'b']
s2  = pd.Series(s2, index=['bf', 'zs', 'c'])
print(s2)
print('----')
s2[pd.isnull(s2)] = 0
print(s2)

#自动对齐
s4 = pd.Series(data=[22, 33, 44], index=['s1', 's2', 's3'])
s5 = pd.Series(data=[22, 33, 99], index=['s3', 's4', 's2'])
print('------')
print(s4 + s5)
print('-----')
#name series对象 index 也有name属性
s6 = pd.Series({'bf': 100, 'zs': 88, 'ls': 99})
s6.name = '数学'
s6.index.name = '考试成绩'
print(s6)


# 注意：使用下标值去切片是前闭后开区间 使用索引值去切片是全闭区间

# Series的运算
# numpy中的数组运算，在Series中都保留了，都可以使用，并且Series在进行数组运算的时候
# 索引与值之间的映射关系不会发生改变
# 注意：其实在操作Series的时候，基本上可以把Series看成numpy中的一维数组来进行

# NaN在pandas中表示一个缺失值或NA值
# pandas中的isnull和notnull两个函数可以用于在Series中检测缺失值，这两个函数返回一个布尔类型的Series
# 可以快速找到缺失值并给其赋一个默认值

# Series自动对齐
# 当多个series对象之间进行运算的时候，如果不同series之间具有不同的索引值，那么运算会
# 自动对齐不同索引值的数据，如果某个series没有某个索引值，那么结果会赋值为NaN

# Series及其索引的name属性
# Series对象本身与其索引都有一个name属性，默认为空，根据需要可以进行赋值操作