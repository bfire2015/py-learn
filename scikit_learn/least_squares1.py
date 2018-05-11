#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/5/8'
"""

#线性回归类
from sklearn.linear_model import LinearRegression
#原始数据 = 训练数据 + 测试数据 数据划分类
from sklearn.model_selection import train_test_split
#数据标准化
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
import time

##设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#加载数据
#日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
path1 = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path1, sep=';', low_memory=False)
# 没有混合类型的时候可以通过low_memory=False调用更多内存，加快效率）

#print(df.head(2))
#print(df.index)
#print(df.columns)

#查看数据结构
#print(df.info())

# 异常数据处理(异常数据过滤)
# 替换非法字符为np.nan
new_df = df.replace('?', np.nan)
# 只要有一个数据为空，就进行行删除操作
datas = new_df.dropna(axis=0, how='any')
# 观察数据的多种统计指标(只能看数值型的 本来9个的变7个了)
#print(datas.describe().T)

# 需求：构建时间和功率之间的映射关系，可以认为：特征属性为时间；目标属性为功率值。
# 获取x和y变量, 并将时间转换为数值型连续变量

# 创建一个时间函数格式化字符串
def date_format(dt):
	#dt显示是一个Series
	t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
	return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

X = datas.iloc[:, 0:2]
#print(X)
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas['Global_active_power']
#print(Y.head(4))
#print(X.head(4))
#print(type(X))
#print(type(Y))

# 对数据集进行测试集、训练集划分
# X：特征矩阵(类型一般是DataFrame)
# Y：特征对应的Label标签或目标属性(类型一般是Series)

# test_size: 对X/Y进行划分的时候，测试集合的数据占比, 是一个(0,1)之间的float类型的值
# random_state: 数据分割是基于随机器进行分割的，该参数给定随机数种子；
# 给一个值(int类型)的作用就是保证每次分割所产生的数数据集是完全相同的
# 默认的随机数种子是当前时间戳 random_state=None的情况下

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)

#查看训练集上数据信息（X）
#print(X_train.describe().T)

# 特征数据标准化（也可以说是正常化、归一化、正规化）
# StandardScaler：将数据转换为标准差为1的数据集(有一个数据的映射)
# scikit-learn中：如果一个API名字有fit，那么就有模型训练的含义，没法返回值
# scikit-learn中：如果一个API名字中有transform， 那么就表示对数据具有转换的含义操作
# scikit-learn中：如果一个API名字中有predict，那么就表示进行数据预测，会有一个预测结果输出
# scikit-learn中：如果一个API名字中既有fit又有transform的情况下，那就是两者的结合(先做fit，再做transform)

# 模型对象创建
ss = StandardScaler()
# 训练模型并转换训练集
#print('转换前：x_train', X_train)
X_train = ss.fit_transform(X_train)
#print('转换后：x_train', X_train)
# 直接使用在模型构建数据上进行一个数据标准化操作 (测试集)
#print('转换前：x_test', X_test)
X_test = ss.transform(X_test)
#print('转换后：x_test', X_test)

#print(X_train.describe().T)
#报错
#print(type(X_train))
#print(X_train.shape, X_train.ndim)
#print(pd.DataFrame(X_train).describe().T)

# 模型训练
# 模型对象构建
# fit_intercept fit训练 intercept截距
# LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
# n_jobs模型训练的任务数 是否并行 并行需要至少2个cpu的 基本没什么用这个参数
lr = LinearRegression(fit_intercept=True)
#训练模型
lr.fit(X_train, Y_train)
#模型校验
#预测结果
y_predict = lr.predict(X_test)

# 回归中的R^2就是准确率
print("训练集上r^2:", lr.score(X_train, Y_train))
print("测试集上r^2:", lr.score(X_test, Y_test))

# 预测值与实际值的差值 平方和 再求均值
mse = np.average((y_predict-Y_test)**2)
rmse = np.sqrt(mse)
print('rmse:',rmse)

# 输出模型训练得到的相关参数
# 注意：第1、2、6个系数为0

print("模型的系数(θ):", end="")
print(lr.coef_)
print("模型的截距:", end='')
print(lr.intercept_)

# 模型保存/持久化（跳过）
# 加载模型 并进行预测（跳过）

# 预测值和实际值画图比较
t = np.arange(len(X_test))
# 建一个画布，facecolor是背景色
plt.figure(facecolor='w')
plt.plot(t, Y_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label='预测值')
# 显示图例，设置图例的位置
plt.legend(loc = 'upper left')
plt.title('线性回归预测时间和功率之间的关系', fontsize=20)
#加网格
plt.grid(b=True)
plt.show()


## 功率和电流之间的关系
X = datas.iloc[:,2:4]
Y2 = datas.iloc[:, 5]

## 数据分割
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size=0.2, random_state=0)

## 数据归一化
scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train) # 训练并转换
X2_test = scaler2.transform(X2_test)## 直接使用在模型构建数据上进行一个数据标准化操作

## 模型训练
lr2 = LinearRegression()
lr2.fit(X2_train, Y2_train)##训练模型

##结果预测
Y2_predict = lr2.predict(X2_test)

##模型评估
print('电流预测准确率:', lr2.score(X2_test, Y2_test))
print('电流参数：', lr2.coef_)

## 绘制图表
#### 电流关系
t = np.arange(len(X2_test))
plt.figure(facecolor='w')
plt.plot(t, Y2_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, Y2_predict, 'g-', linewidth=2, label='预测值')
plt.legend(loc = 'lower right')
plt.title('线性回归预测功率与电流之间的关系', fontsize=20)
plt.grid(b=True)
plt.show()

