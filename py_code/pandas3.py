#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/1/11'
"""

# DataFrame的创建：1、通过二维数组
# DataFrame的创建： 2、字典
# 字典中的value只能是一维数组或者单个的简单数据类型，如果是数组长度必须一致
# 索引对象，不管是Series还是DataFrame对象，都有索引对象
# 他们的自动对齐功能也是通过索引实现的
# DataFrame可以直接通过列索引获取指定列的数据
# 如果需要获取指定行的数据的话，需要通过ix方法来获取对应行索引的行数据
# DataFrame可以切片操作
# 修改值  新增列 新增行  numpy是不能加新行新列的 但是DataFrame可以
# 修改某个具体对象的值，即可以先列后行 也可以先行后列 最好是先列后行可以自动改变对象的数据类型

import pandas as pd
import numpy as np

arr = [['bf', 100], ['zs', 88], ['ls', 99]]
df1 = pd.DataFrame(arr)
# print(df1)
# print(df1.index)
# print(df1.columns)
# print(df1.dtypes)
#
# df1 = pd.DataFrame(arr, index=['1line', '2line', '3line'], columns=['name', 'score'])
# print(df1)

dict1 = {
	'yuwen' : [90, 88, 67],
	'shuxue' : [99, 78, 89],
	'english' : [98, 102, 125],
	'physics' : 88
}

df2 = pd.DataFrame(dict1)
df2.index = ['bf', 'zs', 'ls']
print(df2)

#print(df2['bf']['yuwen']) #error
print(df2['yuwen']['bf'])
print(df2.ix['bf']['yuwen'])
print(df2.ix[:2, :2])#切片
print(df2.dtypes)
df2['english'] = np.nan
df2['english'] = [0, 0, 0]
df2.ix['bf'] = [100, 100, 110, 111]
df2['xxx'] = np.nan
df2.ix['yyy'] = np.nan
print(df2.dtypes)