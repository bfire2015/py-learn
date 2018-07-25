#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '70、单标签多分类算法原理_ovo与ovr的区别_笔记2'
__author__ = 'BfireLai'
__mtime__ = '2018/7/25'
"""

from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

#数据获取
iris = datasets.load_iris()
x, y = iris.data, iris.target
print('样本数量%d,特征数量%d'%x.shape)
print(y)

#模型创建
clf = OneVsRestClassifier(LinearSVC(random_state=0))
#模型创建
clf.fit(x, y)

#预测结果输出
print(clf.predict(x))

#模型属性输出
k = 1
for item in clf.estimators_:
	print('第%d个模型'%k)
	print(item)
	k += 1

print(clf.classes_)