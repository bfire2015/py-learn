#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '这个例子说明：如果进行特征数据标准化则需要截距这个参数，否则会出现偏移'
__author__ = 'BfireLai'
__mtime__ = '2018/5/10'
"""
#引入所需要的全部包
from  sklearn.model_selection import  train_test_split #数据划分类
from sklearn.preprocessing import 	StandardScaler #数据标准化

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import time


##设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#加载数据
#日期、时间、有效功率、无效功率、电压、电流、厨房用电功率、洗衣机用电功率、热水器用电功率
path1 = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path1, sep=';', low_memory=False)
# 没有混合类型的时候可以通过low_memory=False调用更多内存，加快效率）

X2 = df.iloc[:, 2:4]
Y2 = df.iloc[:,5]
#print(X2.shape)

#数据分割
X2_train,X2_test,Y2_train,Y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=0)

#模型对象创建
ss = StandardScaler()
#训练模型并转换训练集
X2_train = ss.fit_transform(X2_train)
#直接使用在模型构建数据上进行一个数据标准化操作（测试集）
X2_test = ss.transform(X2_test)

#将X和Y 转换为矩阵的形式
X = np.mat(X2_train)
Y = np.mat(Y2_train).reshape(-1, 1)

#计算Q
theta = (X.T * X).I * X.T * Y
#print(theta)

y_hat = np.mat(X2_test) * theta

#画图看看
##电流关系
t = np.arange(len(X2_test))
plt.figure(facecolor='w')
plt.plot(t, Y2_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, y_hat, 'g-', linewidth=2, label='预测值')
plt.legend(loc = 'lower right')
plt.title('线性回归预测功率与电流之间的关系', fontsize=20)
plt.grid(b=True)
plt.show()


##模型保存、持久化
#在机器学习部署的时候，实际上其中一种方式就是将模型进行输出
#另个一种方式就是直接将预测结果输出数据库
#模型输出一般是将模型输出到磁盘文件
from sklearn.externals import joblib

#保存模型要求给定的文件所在的文件夹必须存在
joblib.dump(ss, 'datas/data_ss.model')##将标准化模型保存
# joblib.dump(lr,'datas/data_lr.model')##将模型保存

#加载模型
ss3 = joblib.load('datas/data_ss.model')##加载模型
#使用加载的模型进行预测
data2 = [[12, 17]]
data2= ss3.transform(data2)
#print(data2)




