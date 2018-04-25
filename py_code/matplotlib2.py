#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/1/11'
"""

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 8))
labels = ['part one', 'part two', 'part three']
data = [60, 30, 10]
colors = ['red', 'yellow', 'blue']
explode = [0, 0.05, 0]
plt.pie(data, explode=explode, labels=labels, colors=colors, labeldistance=1.1, startangle=90, autopct='%.2f%%', pctdistance=0.5, shadow=False)
plt.axis('equal')
plt.legend()
plt.show()