#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/1/11'
"""
# 带e的科学计数法
# 4.773e-101是科学记数的写法，就是4.773X10^-101的意思，即4.773乘以10的-101次方
# f表示float 1表示小数点后一位小数

#labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
#autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
#shadow，饼是否有阴影
#startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
#pctdistance，百分比的text离圆心的距离

# 设置x，y轴刻度一致，这样饼图才能是圆的
#plt.axis('equal')

# 显示图例
#plt.legend()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

#生成数据
dataOut = np.arange(24).reshape((4,6))
#print(dataOut)

#保存数据 科学记数法  带e 4.773e-101
np.savetxt('data.txt', dataOut, fmt='%.2f')

#读取数据
data = np.loadtxt('data.txt')
#print(data, data.dtype)

y = np.random.randint(1, 11, 5)
print(y)
x = np.arange(len(y))
print(x)

plt.figure(figsize=(6, 8))
print(rcParams['figure.figsize'])
# plt.plot(x, y, color='r')
# plt.bar(x, y, color='g')
plt.pie(y, explode=[0, 0.2, 0, 0, 0])
plt.show()




