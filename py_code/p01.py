# -*- coding: utf-8 -*-
# raw_input print 是最基本的输入与输出
#name = raw_input('please enter your name:')
#print ('hello,', name)

##numpy  fortest
import numpy as np

print('np.arange10:', np.arange(10))
for i in range(10):
	print(i)

a = np.arange(10)
print('a^2:', a**2)


##scipy  fortest
from scipy import linalg
b = np.array([[1, 2], [30, 40]])
print(b)
# 二阶方阵行列式
print(linalg.det(b))
# 推荐用scipy.linalg代替numpy.linalg

##pandas fortest
import pandas as pd

s = pd.Series([2, 4, 5, np.nan, 8, 9])
print(s)

dates = pd.date_range('20171222', periods=7)
print(dates)

df = pd.DataFrame(np.random.rand(7, 4), index=dates, columns=list('ABCD'))
print(df)
# 转置
# print(df.T)

print(df.sort_values(by='B'))

