#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/1/10'
"""

#11、pandas简介及其数据结构Series详解_笔记
# 熊猫( panda的名词复数 )
# pandas 是基于NumPy 的一种工具，该工具是为了解决数据分析任务而创建的。
# Pandas 纳入了大量库和一些标准的数据模型，提供了高效地操作大型数据集所需的工具。
# pandas提供了大量能使我们快速便捷地处理数据的函数和方法。
# 你很快就会发现，它是使Python成为强大而高效的数据分析环境的重要因素之一。

# pandas 依赖python库：setuptools numpy python-dateutil  pytz
# pandas 的三大作用：数据的引入 数据的特征提取  数据的清洗

# pandas的基本数据结构：Series和DataFrame
# Series：一种类似于一维数组的对象，是由一组数据（各种numpy数据类型）以及一组与之相关的
# 数据标签（即索引）组成。仅由一组数据也可产生简单的Series对象，注意：Series中的索引值是可以重复的

# DataFrame数据框架：一个表格型的数据结构，包含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型等）
# DataFrame既有行索引也有列索引，可以被看作是由Series组成的字典

# 第一种：Series通过一维数组创建

# 通过数组创建Series的时候，如果没有为数据指定索引的话，会自动创建一个从0到n-1的整数索引
# 当Series对象创建好后，可以通过index修改索引值

# 明确给定索引值与数据类型

# 第二种：Series通过字典的方式创建

import numpy as np
import pandas as pd

#arr = np.random.randint(10, 20, (3,4))
#print('np->randint:', arr)

arr = np.array([22, 33, np.nan, 90])
#print(arr, arr.dtype)
s1 = pd.Series(arr)
# print(s1)
# print('series的属性')
# print(s1.dtype)
# print(s1.index)
# print(s1.values)

# s1.index = [u'bfire', 'a2', 'a3', 'a4']
# print(s1)

s2 = pd.Series(data=[88, 99, 100], index=['语文', '数学', '英语'], dtype=np.float)
print(s2)
print('------')
dict1 = {'语文':77, '数学':88, '英语':99}
s3 = pd.Series(dict1, dtype=np.float)
print(s3)