#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '多项式扩展与过拟合问题'
__author__ = 'BfireLai'
__mtime__ = '2018/5/10'
"""

#线性回归类
from sklearn.linear_model import LinearRegression
#原始数据 = 训练数据+ 测试数据
#数据划分类
from sklearn.model_selection import train_test_split
#数据标准化
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import  numpy as np
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
#日期、时间、有效功率、无效功率、电压、电流、厨房用电功率、洗衣机用电功率、热水器用电功率
path1 = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path1, sep=';', low_memory=False)
# 没有混合类型的时候可以通过low_memory=False调用更多内存，加快效率）

# 查看数据结构
# print(df.info())

# 异常数据处理(异常数据过滤)
# 替换非法字符为np.nan
new_df = df.replace('?', np.nan)
# 只要有一个数据为空，就进行行删除操作
datas = new_df.dropna(axis=0, how='any')
# 观察数据的多种统计指标(只能看数值型的 本来9个的变7个了)
# print(datas.describe().T)


# 需求：构建时间和功率之间的映射关系，可以认为：特征属性为时间；目标属性为功率值。
# 获取x和y变量, 并将时间转换为数值型连续变量

# 创建一个时间函数格式化字符串
def date_format(dt):
    # dt显示是一个Series
    # print(dt.index)
    # print(dt)
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

X = datas.iloc[:, 0:2]
# print(X)
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
# Y = datas['Global_active_power']
Y = datas['Voltage']
# Y = datas[4].values
# print(Y.head(4))
# print(X.head(4))
# print(type(X))
# print(type(Y))


# 对数据集进行测试集、训练集划分
# X：特征矩阵(类型一般是DataFrame)
# Y：特征对应的Label标签或目标属性(类型一般是Series)

# test_size: 对X/Y进行划分的时候，测试集合的数据占比, 是一个(0,1)之间的float类型的值
# random_state: 数据分割是基于随机器进行分割的，该参数给定随机数种子；
# 给一个值(int类型)的作用就是保证每次分割所产生的数数据集是完全相同的
# 默认的随机数种子是当前时间戳 random_state=None的情况下
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

# 查看训练集上的数据信息(X)
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
X_train = ss.fit_transform(X_train)
# 直接使用在模型构建数据上进行一个数据标准化操作 (测试集)
X_test = ss.transform(X_test)

# print(X_train.describe().T)
# print(type(X_train))
# print(X_train.shape, X_train.ndim)
#print(pd.DataFrame(X_train).describe().T)



# 多项式扩展（多项式曲线拟合）：将特征与特征之间进行融合，从而形成新的特征的一个过程；
# 从数学空间上来讲，就是将低维度空间的点映射到高维度空间中。
# 更容易找到隐含的特性  属于特征工程的某一种操作  实际中用得比较少，一般用高斯扩展比较多 后面会讲

# 作用：通过多项式扩展后，我们可以提高模型的准确率或效果

# 过拟合：模型在训练集上效果非常好，但是在测试集中效果不好

# 多项式扩展的时候，如果指定的阶数比较大，那么有可能导致过拟合

# 从线性回归模型中，我们可以认为训练出来的模型参数值越大，就表示越存在过拟合的情况



## 时间和电压之间的关系(Linear-多项式)
# Pipeline：管道的意思，讲多个操作合并成为一个操作
# Pipleline总可以给定多个不同的操作，给定每个不同操作的名称即可，执行的时候，按照从前到后的顺序执行
# Pipleline对象在执行的过程中，当调用某个方法的时候，会调用对应过程的对应对象的对应方法
# eg：在下面这个案例中，调用了fit方法，
# 那么对数据调用第一步操作：PolynomialFeatures的fit_transform方法对数据进行转换并构建模型
# 然后对转换之后的数据调用第二步操作: LinearRegression的fit方法构建模型
# eg: 在下面这个案例中，调用了predict方法，
# 那么对数据调用第一步操作：PolynomialFeatures的transform方法对数据进行转换
# 然后对转换之后的数据调用第二步操作: LinearRegression的predict方法进行预测
models = [
	Pipeline([
		('Poly', PolynomialFeatures()),#给定进行多项式扩展操作，第一个操作：多项式扩展
		('Linear', LinearRegression(fit_intercept=False)) #第二个操作,线性回归
	])
]
model = models[0]

#模型训练
t = np.arange(len(X_test))
N = 5;
d_pool = np.arange(1, N, 1) #阶
m = d_pool.size
clrs = [] #颜色
for c in np.linspace(16711680, 255, m):
	clrs.append('#%06x'% int(c))

line_width = 3

plt.figure(figsize=(12, 6), facecolor='w')#创建一个绘图窗口，设置大小，设置颜色
for i,d in enumerate(d_pool):
	plt.subplot(N-1, 1, i + 1)
	plt.plot(t, Y_test, 'k-', label='真实值', ms=10, zorder=N)
	##设置管道对象中的参数值，poly是在管道对象中定义的操作名称，
	##后面跟参数名称，中间是两个下划线
	model.set_params(Poly__degree=d)#设置多项式的阶乘
	model.fit(X_train, Y_train) #模型训练
	#linear 是管道中定义的操作名称
	#获取线性回归算法模型对象
	lin = model.get_params()['Linear']
	output = '%d阶，系数为：'% d
	#判断lin对象中是否有对应属性
	if hasattr(lin, 'alpha_'):
		idx = output.find('系数')
		output = output[:idx] + ('alpha=%.6f,'% lin.alpha_) + output[idx:]
	if hasattr(lin, 'l1_ratio_'):
		idx = output.find('系数')
		output = output[:idx] + ('l1_ratio=%.6f,'% lin.l1_ratio_) + output[idx:]
	print(output, lin.coef_.ravel())

	#模型结果预测
	y_hat = model.predict(X_test)
	#计算评估值
	s = model.score(X_test, Y_test)

	#画图
	z = N - 1 if(d == 2) else 0
	label = '%d 阶，准确率=%.3f'%(d, s)
	plt.plot(t, y_hat, color=clrs[i], lw=line_width, alpha=0.75, label=label,zorder=z)
	plt.legend(loc='upper left')
	plt.grid(True)
	plt.ylabel('%d阶结果'%d, fontsize=12)

##预测值和实际值画图比较
plt.suptitle('线性回归预测时间和电压之间的多项式关系 ', fontsize=20)
plt.grid(b=True)
plt.show()
