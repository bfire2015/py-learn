#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '普通最小二乘求线性回归问题 自己模型'
__author__ = 'BfireLai'
__mtime__ = '2018/5/16'
"""

#数据校验
def validate(x, y):
	if len(x) != len(y):
		raise Exception('参数异常')
	else:
		m = len(x[0])
		for l in x:
			if len(l) != m:
				raise Exception('参数异常2')
		if len(y[0]) != 1:
			raise Exception('参数异常3')

#计算差异值
def calcDiffe(x, y, a):
	#计算ax -y
	lx = len(x)
	la = len(a)
	if lx == la:
		result = 0
		for i in range(lx):
			result += x[i] * a[i]
		return y - result
	elif lx +1 == la:
		result = 0
		for i in range(lx):
			result += x[i] * a[i]
		result += 1* a[lx] #加上常数项
		return  y - result
	else:
		raise Exception('参数异常4')


def fit(x, y, alphas, threshold=1e-6, maxlter=200, addConstantitem=True):
	import math
	import numpy as np
	##检验
	validate(x, y)
	##构建模型
	l = len(alphas)
	m = len(y)
	n = len(x[0] + 1 if addConstantitem else len(x[0]))#样本个数
	b = [True for i in range(l)] #模型格式，控制最优模型
	##差异性
	J = [np.nan for i in range(l)] ##loss 函数的值
	#1\随机初始化0值，a的最后一列为常数项
	a = [[0 for j in range(n)] for i in range(l)] #theta 是模型的系数
	#2\开始计算
	for times in range(maxlter):
		for i in range(l):
			if not b[i]:
				#如果当前alpha值计算得到最优解，则不继续计算
				continue

			ta = a[i]
			for j in range(n):
				alpha = alphas[i]
				ts = 0
				for k in range(m):
					if j == n -1 and addConstantitem:
						ts += alpha * calcDiffe(x[k], y[k][0], a[i])*1
					else:
						ts += alpha * calcDiffe(x[k], y[k][0], a[i])* x[k][j]
				t = ta[j] + ts
				ta[j] = t
			##计算完一个alpha 值的0损失函数
			flag = True
			js = 0
			for k in range(m):
				js += math.pow(calcDiffe(x[k], y[k][0], a[i]), 2) + a[i][j]
				if js > J[i]:
					flag = False
					break
			if flag:
				J[i] = js
				for j in range(n):
					a[i][j] = ta[j]
			else:
				##标记当前alpha 值不需要计算
				b[i] = False

		#计算完一个迭代，当目标函数、损失函数值有一个小于threshold结束循环
		r = [0 for j in J if j <= threshold]
		if len(r) > 0:
			break;

	#3\获取最优的alphas 值以及对应的0值
	min_a = a[0]
	min_j = J[0]
	min_alpha = alphas[0]
	for i in range(l):
		if J[i] < min_j:
			min_j = J[i]
			min_a = a[i]
			min_alpha = alphas[i]

	print('最优alpha值=',min_alpha)

	#4\返回最终0值
	return min_a


#预测结果
def predict(X, a):
	y = []
	n = len(a) -1
	for x in X:
		result = 0
		for i in range(n):
			result += x[i] * a[i]
		result += a[n]
		y.append(result)
	return y

#计算实际值与预测值之间的相关性
def calcRScore(y, py):
	if len(y) != len(py):
		raise Exception('参数异常')
	import math
	import numpy as np
	avgy = np.average(y)
	m = len(y)
	rss = 0.0
	tss = 0
	for i in range(m):
		rss += math.pow(y[i] - py[i], 2)
		tss += math.pow(y[i] - avgy, 2)
	r = 1.0 - 1.0 * rss/tss
	return r


##下面实际数据产生、训练、与分析
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,LassoCV,RidgeCV,ElasticNetCV

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#创建模拟数据
np.random.seed(0)
np.set_printoptions(linewidth=1000, suppress=True)
N = 10
x = np.linspace(0, 6, N) + np.random.randn(N)
y = 1.8*x**3 + x**2 - 14*x - 7 + np.random.randn(N)
x.shape = -1,1
y.shape = -1,1
print(x)

##模拟数据产生
x_hat = np.linspace(x.min(), x.max(), num=100)
x_hat.shape = -1,1

#线性模型
model = LinearRegression()
model.fit(x, y)
y_hat = model.predict(x_hat)
s1 = calcRScore(y, model.predict(x))
print(model.score(x, y))##r^2输出
print('模块自带实现：---------')
print('参数列表：', model.coef_)
print('截距：', model.intercept_)

##自模型
ma = fit(x, y, np.logspace(-4, -2, 100), addConstantitem=True)
y_hat2 = predict(x_hat, ma)
s2 = calcRScore(y, predict(x, ma))
print('自定义实现模型--------')
print('参数列表', ma)

#画图
plt.figure(facecolor='w')
plt.plot(x, y, 'ro', ms=10, zorder=3)
plt.plot(x_hat, y_hat, color='b', lw=2, alpha= 0.75, label='py模型,$r^2$=%.3f'%s1, zorder=1)
plt.plot(x_hat, y_hat2, color='r', lw=2, alpha= 0.75, label='自己模型,$r^2$=%.3f'%s2, zorder=2)
plt.legend(loc='upper left')
plt.grid(True)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.suptitle('自定义的线性模型和模块中的线性模型比较', fontsize=22)
plt.show()


