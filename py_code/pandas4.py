#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/1/11'
"""

# pandas的基本功能
# 1、数据文件读取 文本数据读取
# 2、索引、选取和数据过滤
# 3、算术运算和数据对齐
# 4、函数的应用和映射
# 5、重置索引

# 通过pandas提供的read_XXX相关的函数可以读取文件中的数据，并形成DataFrame
# 常用的数据读取方法为read_csv 主要可以读取文本类型的数据
# help(pd.read_csv)
# help(pd.read_excel)
# csv的数据的表格形式的 并且是最简单的表格 直接看看图片

# pandas：数据过滤获取
# 通过DataFrame的相关方式可以获取对应的列或者数据形成一个新的DataFrame，方便后续进行统计计算
# pandas中缺省值NaN处理方法：1 isnull  2 notnull  3 dropna  4 fillna
# df.dropna()  默认丢弃只要包含nan数据的行 axis=1则是丢弃列 how='any'默认 如果设置how='all'则表示全部为nan才丢弃

# df.fillna()  填充缺失值

# pandas常用的数学统计方法
# count 计算非NA值的数量
# describe 针对Series或DataFrame列计算统计
# min/max/sum 计算最小值 最大值  总和
# argmin argmax 计算能够获取到最小值和最大值的索引位置（整数）
# idxmin idxmax 计算能够获取到最小值和最大值的索引值
# quantile   计算样本的分位数（0到1）
# mean   值的平均数
# median  值的中位数
# mad 根据平均值计算平均绝对距离差
# var 样本数值的方差
# std 样本值的标准差
# cumsum 样本值的累计和
# cummin  cummax  样本的累计最小值 最大值
# cumprod  样本值的累计积
# pct_change  计算百分数变化
# print(df2.describe())
# print(df2.quantile())


# 相关系数   具体看图片
# print(df2.corr())
# 协方差
# print(df2.cov())

# pandas:唯一值、值频率计算以及成员资格
# unique方法用于获取Series或DataFrame某列中的唯一值数组（去重数据后的数组)
# value_counts方法用于计算一个Series或DataFrame某列中各值的出现频率
# isin方法用于判断矢量化集合的成员资格，是否在里面，可用于选取Series中或DataFrame列中数据的子集

import pandas as pd
import numpy as np

# pd.read_csv()
# pd.read_excel()
# pd.read_json()
# DataFrame数据的切片
# pandas当中处理NaN缺省值的方式：1、isnull 2 notnull
# 3、dropna()  4、fillna()

dict1 = {
	'yuwen' : [99, 88, 66],
	'shuxue' : [99, 77, 89],
	'english' : [98, 102, 125],
	'physics' : 88
}

df2 = pd.DataFrame(dict1)
print('----')
# print(df2.dropna(axis=1))

df3 = pd.DataFrame(np.random.random((7, 3)))
print(df3)
print('-----')
# print(df3.fillna(1))
# print(df3.fillna({1: 0.5, 2: -1}))

# print(df2.describe())
# print(df2.median())
# print(df2.var())
# print(df2.std())
# print(df2.corr())
# print(df2.cov())

s1 = pd.Series(['a', 'b', 'c', 'b', 'a'])
print(s1.unique())
print(s1.value_counts()['a'])
print(s1.isin(['a', 'b']))

df4 = pd.DataFrame(np.random.randint(10, 16, (3, 3)), columns=['bf', 'zs', 'ls'])
print(df4)
# print(df4.ix[0].unique())
# print(df4['bf'].unique())
# print(df4['bf'].value_counts())
# print(df4.ix[0].value_counts())
print(df4['bf'].isin([11]))

