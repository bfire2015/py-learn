#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '70、单标签多分类算法原理_ovo与ovr的区别_笔记'
__author__ = 'BfireLai'
__mtime__ = '2018/7/25'
"""

from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

#加载数据
iris = datasets.load_iris()

#获取x,y
x,y = iris.data, iris.target
print('样本数量：%d,特征数量:%d'%x.shape)
print(y)

#模型构建
clf = OneVsOneClassifier(LinearSVC(random_state=0))
#模型训练
clf.fit(x, y)

#输出预测结果值
print(clf.predict(x))

#模型属性输出
k = 1
for item in clf.estimators_:
	print("第%d个模型"%k)
	print(item)
	k += 1
print(clf.classes_)