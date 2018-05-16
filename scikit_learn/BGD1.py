#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '线性回归总结与BGD算法代码实现'
__author__ = 'BfireLai'
__mtime__ = '2018/5/15'
"""

#实现一个批量梯度下降算法求解线性回归问题模型

import math
import numpy as np

def validate(x, y):
	#检验x ,y的格式是否正确
	if len(x) != len(y):
		raise Exception('样本参数异常')
	else:
		n = len(x[0])
		for l in x:
			if len(l) != n:
				raise Exception('参数异常')
		if len(y[0]) != 1:
			raise Exception('参数异常2')

def predict(x, theta, intercept=0.0):
	#算出预测值
	#x 一条样本数据
	#theta 参数向量
	#intercept 截距
	#返回值是 y的预测结果
	result = 0.0
	#1\ x与q的相乘
	n = len(x)
	for i in range(n):
		result += x[i] * theta[i]
	#2\加上截距值
	result += intercept
	#3\返回结果
	return result

def predict_X(x, theta, intercept=0.0):
	#预测y[]
	#x, 是一个二维矩阵
	Y = []
	for i in x:
		Y.append(predict(i, theta, intercept))
	return Y

def fit(x, y, alpha=0.01, max_iter=100, fit_intercept=True, tol=1e-4):
	#进行模型的训练，返回模型参数q值与截距
	#x 输入的特征性的矩阵 二维m*n m表示样本个数，n是x的维度数目
	#y 输入的目标属性矩阵 二维m*k m表示样本个数，k表示y值的个数，一般k=1,目前这个阶段就考虑一个y值
	#alpha 学习率 步长 默认0.01
	#max_iter 最大迭代次数 默认100
	#fit_intercept 是否训练截距 默认Treu
	#tol 线性回归当中的平方和损失函数的误差值 如果小于给定tol值 就结束迭代退出循环， 默认1e-4
	#return (theta, intercept)

	#1\检验x,y数据格式是否正确
	validate(x, y)

	#2\开始训练模型 迭代计算参数
	#获取行和列 分别记做样本个数m 和特征属性个数n
	m, n = np.shape(x)
	#定义需要训练的参数 并且初始化
	theta = [0 for i in range(n)]
	intercept = 0

	#定义一个临时的变量
	diff = [0 for i in range(m)]
	max_iter = 100 if max_iter <= 0 else max_iter

	#开始进行迭代 更新参数
	for i in range(max_iter):
		#在当前q的取值情况下，预测值与实际值的差值
		for k in range(m):
			y_true = y[k][0]
			y_predict = predict(x[k], theta, intercept)
			diff[k] = y_true - y_predict
		#对q进行更新
		for j in range(n):
			#计算梯度值
			gd = 0
			for k in range(m):
				gd += diff[k] * x[k][j]
			theta[j] += alpha * gd
		#训练截距（相当于求解q的时候，对应维度上x的取值是1）
		if fit_intercept:
			gd = np.sum(diff)
			intercept += alpha * gd

		#需要判断损失函数是否已经收敛
		#1、计算损失函数值
		#2、判断损失函数值与给定的tol
		sum_j = 0.0
		for k in range(m):
			y_true = y[k][0]
			y_predict = predict(x[k], theta, intercept)
			j = y_true - y_predict
			sum_j += math.pow(j, 2)
		sum_j /= m

		if sum_j < tol:
			break

	#3\返回参数
	return (theta, intercept)

def score_x_y(x, y, theta, intercept=0.0):
	#1\先要得到预测值
	y_predict = predict_X(x, theta, intercept)
	return score(y, y_predict)

def score(y, y_predict):
	#计算回归模型r^2 值
	#1\计算rss 与 tss
	average_y = np.average(y)
	m = len(y)
	rss = 0.0
	tss = 0.0
	for k in range(m):
		rss += math.pow(y[k] - y_predict[k], 2)
		tss += math.pow(y[k]- average_y, 2)
	#2\计算r^2的值
	r_2 = 1.0 - 1.0 * rss/tss
	#3\返回
	return r_2

#测试BGD 与 scikit-learn 最小二乘法比较
import  numpy as np
import  matplotlib as mpl
import  matplotlib.pyplot as plt
from sklearn.linear_model import  LinearRegression

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#创建数据（样本数据）
np.random.seed(0)
np.set_printoptions(linewidth=100, suppress=True)
N = 10
x = np.linspace(0, 6, N) + np.random.randn(N)
y = 1.8*x **3 + x**2 - 14*x - 7 + np.random.randn(N)
x.shape = -1,1
y.shape = -1,1
print(x)
print(y)
#在样本的基础上 进行模型训练（最小二乘法、梯度下降法）
lr = LinearRegression(fit_intercept=True)
lr.fit(x, y)
print('scikit-learn最小二乘法实现：')
s1 = score(y, lr.predict(x))
print('自己r^2值计算：%.5f'%s1)
print('框架r^2值计算：%.5f'%lr.score(x, y))
print('参数q',lr.coef_)
print('截距:', lr.intercept_)

#自己的BGD训练
theta,intercept = fit(x, y, alpha=0.01, max_iter=100, fit_intercept=True)
print('自己的BGD梯度下降法实现：')
s2 = score(y, predict_X(x, theta, intercept))
print('bgd自己r^2值计算：%.5f'%s2)
print('参数q:',theta)
print('截距:', intercept)

#画图
plt.figure(figsize=(12, 6), facecolor='w')
#为了画那条直线，需要产生很多模拟数据
x_hat = np.linspace(x.min(), x.max(), num=100)
x_hat.shape = -1,1
#框架里的最小二乘法
y_hat = lr.predict(x_hat)
#自己bgd 梯度下降法
y_hat2 = predict_X(x_hat, theta, intercept)

plt.plot(x, y, 'ro', ms =10, zorder=3)
plt.plot(x_hat, y_hat, color='g', lw=2, alpha=0.75, label='普通最小二乘法,准确率$r^2:$%.3f'%s1, zorder=2)
plt.plot(x_hat, y_hat2, color='b', lw=2, alpha=0.75, label='bgd梯度下降法，准确率$r^2:$%.3f'%s2, zorder=1)

plt.legend(loc='upper left')
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.suptitle('普通最小二乘法与BGD梯度下降法的线性回归模型比较', fontsize=22)
plt.show()