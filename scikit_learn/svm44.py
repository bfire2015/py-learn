#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '69、SVM算法4个实战综合案例_笔记4'
__author__ = 'BfireLai'
__mtime__ = '2018/7/25'
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV

# 如果模型的解释性要求比较强，就是对返回y值可以做一个解释，一般用：决策树和逻辑回归比较多
# 如果仅仅是要求分类效果比较好，不考虑解释性，一般用：svm比较多
# 居中的话就是KNN 集成算法是对效果来提升的

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

x,y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
##偏移处理
x += 2 * rng.uniform(size=x.shape)
linearly_separable = (x, y)

#月牙数据 圆形 线性（加了偏移的 抖动的）
datasets = [make_moons(noise=0.3, random_state=0), make_circles(noise=0.2, factor=0.4, random_state=1), linearly_separable]

##建模环节，用list 把所有算法装起来
names = ['nearest neighbors', 'logistic', 'decision tree', 'random forest', 'adaboost', 'gbdt', 'svm']
classifiers = [
	KNeighborsClassifier(3),
	LogisticRegressionCV(),
	DecisionTreeClassifier(max_depth=5),
	RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	AdaBoostClassifier(n_estimators=10, learning_rate=1.5),
	GradientBoostingClassifier(n_estimators=10, learning_rate=1.5),
	svm.SVC(C=1, kernel='rbf')
]

##画图
figure = plt.figure(figsize=(27, 9), facecolor='w')
i = 1;
h = .02 #步长
for ds in datasets:
	x,y = ds
	x = StandardScaler().fit_transform(x)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4)
	x_min, x_max = x[:, 0].min() - .5, x[:,0].max() + .5
	y_min, y_max = x[:, 1].min() - .5, x[:,1].max() + .5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	cm =plt.cm.RdBu
	cm_bright = ListedColormap(['r', 'b', 'y'])

	ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
	ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright)
	ax.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap=cm_bright, alpha=0.6)
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xticks(())
	ax.set_yticks(())
	i += 1

	##画每个算法图
	for name, clf in zip(names, classifiers):
		ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
		clf.fit(x_train, y_train)
		score = clf.score(x_test, y_test)
		##hasattr 是判断某个模型中，有没有哪个参数
		##判断 clf 模型， 有没有decision_function
		#np.c_ 让内部数据按列合并
		if hasattr(clf, 'decision_function'):
			z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
		else:
			z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]

		z = z.reshape(xx.shape)
		ax.contourf(xx, yy, z, cmap=cm, alpha=.8)
		ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright)
		ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xticks(())
		ax.set_yticks(())
		ax.set_title(name)
		ax.text(xx.max() - .3, yy.min() + .3, ('%.2f'%score).lstrip('0'), size=25, horizontalalignment='right')
		i += 1


figure.subplots_adjust(left=.02, right=.98)
plt.suptitle('机器学习中各种分类算法模型比较', fontsize=18, color='r')
plt.show()

