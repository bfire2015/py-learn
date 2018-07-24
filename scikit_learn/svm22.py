#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '69、SVM算法4个实战综合案例_笔记2'
__author__ = 'BfireLai'
__mtime__ = '2018/7/24'
"""

#图片识别用深度学习居多 传统的机器学习也可以做到
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

##加载数字图片数据
digits = datasets.load_digits()
print(digits)
##images data target target_names descr
#images 图片的像素点  类似前面的聚类算法图片压缩 黑白图是数组2维
#target 相当于y值 实际值
print(digits.images[10])
print(digits.target[10])

#获取样本数量，并将图片数据格式化
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
print(data.shape)

##模型构建
classifier = svm.SVC(gamma=0.001) #默认是rbf
#也可以使用knn
##取前一半数据训练，后一半数据测试
classifier.fit(data[:int(n_samples/2)], digits.target[:int(n_samples/2)])

##测试数据部分实际值和预测值获取
##后一半数据作为测试集
expected = digits.target[int(n_samples/2):] ##y_test
predicted = classifier.predict(data[int(n_samples/2):]) ##y_predicted
##计算准确率
##classification_report
print('分类器%s的分类效果：\n%s\n'%(classifier, metrics.classification_report(expected, predicted)))
print('混淆矩阵为\n%s'%metrics.confusion_matrix(expected, predicted))
print('score_svm:\n%f'%classifier.score(data[int(n_samples/2):], digits.target[int(n_samples/2):]))

##进行图片展示
plt.figure(facecolor='gray', figsize=(12, 7))
images_and_predictions = list(zip(digits.images[int(n_samples/2):][expected != predicted], expected[expected!=expected], predicted[expected != predicted]))
##通过enumerate ,分别拿出x值，y值 和y的预测的前五个，并且画图
for index, (image, expection, prediction) in enumerate(images_and_predictions[:5]):
	plt.subplot(2, 5, index + 1)
	plt.axis('off')
	plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest') #把cmap中灰度值与image矩阵对应
	plt.title('预测值、实际值：%i/%i'%(prediction, expection))

##画出5个预测成功
images_and_predictions = list(zip(digits.images[int(n_samples/2):][expected==predicted], expected[expected==predicted], predicted[expected==predicted]))

for index, (image, expection, prediction) in enumerate(images_and_predictions[:5]):
	plt.subplot(2, 5, index + 6)
	plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	plt.title('预测值/实际值：%i%i'%(prediction, expection))

plt.subplots_adjust(.04, .02, .97, .94, .2)
plt.suptitle('手写数字的识别', fontsize=18, color='r')
plt.show()
