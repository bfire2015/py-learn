#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '34、使用决策树的可视化工具画出树结构_笔记'
__author__ = 'BfireLai'
__mtime__ = '2018/5/21'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from sklearn.tree import DecisionTreeClassifier #分类树
from sklearn.model_selection import train_test_split #测试集和训练集
from sklearn.feature_selection import SelectKBest #特征选择
from sklearn.feature_selection import chi2 #卡方统计量
from sklearn.preprocessing import MinMaxScaler #数据归一化
from sklearn.decomposition import PCA #主要成分分析

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = u'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore', category=FutureWarning)

iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature_C = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

#读取数据
path = './datas/iris.data'
data = pd.read_csv(path, header=None)
x=data[list(range(4))]#获取X变量
y=pd.Categorical(data[4]).codes#把Y转换成分类型的0,1,2
print("总样本数目：%d;特征属性数目:%d" % x.shape)

print(data.head(5))

#数据进行分割（训练数据和测试数据）
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=14)

x_train, x_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))
y_train = y_train.astype(np.int)
y_test = y_test.astype(np.int)

ss = MinMaxScaler()
#用标准化方法对数据进行处理并转换
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
print ("原始数据各个特征属性的调整最小值:",ss.min_)
print ("原始数据各个特征属性的缩放数据值:",ss.scale_)

#特征选择：从已有的特征中选择出影响目标值最大的特征属性
#常用方法：{ 分类：F统计量、卡方系数，互信息mutual_info_classif
        #{ 连续：皮尔逊相关系数 F统计量 互信息mutual_info_classif
#SelectKBest（卡方系数）

# ch2 = SelectKBest(chi2,k=3)#在当前的案例中，使用SelectKBest这个方法从4个原始的特征属性，选择出来3个
# #K默认为10
# #如果指定了，那么就会返回你所想要的特征的个数
# x_train = ch2.fit_transform(x_train, y_train)#训练并转换
# x_test = ch2.transform(x_test)#转换
#
# select_name_index = ch2.get_support(indices=True)
# print ("对类别判断影响最大的三个特征属性分布是:",ch2.get_support(indices=False))
# print(select_name_index)

#降维：对于数据而言，如果特征属性比较多，在构建过程中，会比较复杂，这个时候考虑将多维（高维）映射到低维的数据
#常用的方法：
#PCA：主成分分析（无监督）
#LDA：线性判别分析（有监督）类内方差最小，人脸识别，通常先做一次pca

# pca = PCA(n_components=2)#构建一个pca对象，设置最终维度是2维
# # #这里是为了后面画图方便，所以将数据维度设置了2维，一般用默认不设置参数就可以
#
# x_train = pca.fit_transform(x_train)#训练并转换
# x_test = pca.transform(x_test)#转换

#模型的构建
model = DecisionTreeClassifier(criterion='entropy',random_state=0, min_samples_leaf=5)#另外也可选gini
#模型训练
model.fit(x_train, y_train)
#模型预测
y_test_hat = model.predict(x_test)


#模型结果的评估
y_test2 = y_test.reshape(-1)
result = (y_test2 == y_test_hat)
print ("准确率:%.2f%%" % (np.mean(result) * 100))
#实际可通过参数获取
print ("Score：", model.score(x_test, y_test))#准确率
print ("Classes:", model.classes_)


# 方式一：输出形成dot文件，然后使用graphviz的dot命令将dot文件转换为pdf
# from sklearn import tree #决策树
# with open('./datas/iris111.dot', 'w') as f:
#     # 将模型model输出到给定的文件中
#     f = tree.export_graphviz(model, out_file=f)
# 命令行执行dot命令： dot -Tpdf iris111.dot -o iris111.pdf

# 方式二：直接使用pydotplus插件生成pdf文件
# from sklearn import tree
# import pydotplus
# dot_data = tree.export_graphviz(model, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("./datas/iris222.pdf")
# graph.write_png("./datas/iris222.png")
#
# 方式三：直接生成图片
from sklearn import tree
# from IPython.display import Image
import pydotplus
dot_data = tree.export_graphviz(model, out_file=None,
                         # feature_names=['特征一', '特征二'],
                         feature_names=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'],
                         class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
#print(dot_data)
# Image(graph.create_png())
graph.write_png("./datas/iris333.png")