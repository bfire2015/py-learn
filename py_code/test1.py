#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2017/12/25'
"""
print('this is test1')

#%d	整数
#%f	浮点数
#%s	字符串
#%x	十六进制整数
#print('这是数字：%d,这是浮点数：%f,这是字符串：%s.' % (11,22,'天下为公'))

print('if and else')

age = 7
if age >= 18:
		print('adult')
elif age >= 6:
	print('geenager')
else:
	print('kid')

print('循环:')
names = ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
for name in names:
	print(name)

print('sum:')
sum = 0
for x in range(101):
	sum = sum + x
print('range(101)就可以生成0-100的整数序列，计算如下：',sum)

print('字典操作：set,add,remove')
s = set([1, 2, 3, 4])
print('s=',s)
s.add(5)
print('s.add=5',s)
s.remove(4)
print('s.remove(4)=',s)
#s.sort()
#print('s.sort()=', s)

f = 'abc'
g = f.replace('a','A')
print('f.replace(abc)',f, g)

print('切片操作:')
h = ['66', '77', '88', '99', '100']
print('h=', h)
r = []
n = 3
for i in range(n):
	r.append(h[i])
print('r.addend(0-3)=',r)
print('h[0:3]=', h[0:3])
print('h[:3]=', h[:3])
print('h[1:3]=', h[1:3])
print('h[-2:]=', h[-2:])
print('h[-2:-1]=',h[-2:-1])

i = range(100)
print('i = range(100)==>', i)
print('i[-10:]=', i[-10:])
print('i[10:20]=', i[10:20])
print('i[:10:2]=', i[:10:2])

#如果给定一个list或tuple，我们可以通过for循环来遍历这个list或tuple，这种遍历我们称为迭代（Iteration）
print('迭代器：')
from collections import Iterable
print('isinstance("abc", Iterable)=', isinstance('abc', Iterable))
print('isinstance([1,2,3], Iterable)=', isinstance([1, 2, 3], Iterable))
print('isinstance(123, Iterable)=', isinstance(123, Iterable))

#列表生成式即List Comprehensions，是Python内置的非常简单却强大的可以用来创建list的生成式。
print('list:')
j = range(1, 11)
print('j=range(1,11)==>', j)

k = []
for x in range(1, 11):
	k.append(x * x)
print('k[i*i]=',k)
print('x[1,11),x%2==',[x * x for  x in range(1, 11) if x%2 == 0])
print('m+n for ABC in XYZ=', [m + n for m in 'ABC' for n in 'XYZ'])

#在Python中，这种一边循环一边计算的机制，称为生成器（Generator）。
print('生成器:')
def fib(max):
	n, a, b = 0, 0, 1
	while n < max:
		yield b
		a, b = b, a + b
		n = n +1

print('斐波拉契数列:fib(6)=', fib(6))

#函数式编程的一个特点就是，允许把函数本身作为参数传入另一个函数，还允许返回一个函数！
print('函数式编程：')
print('1、map()=')
l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print('map(str, [1-9])=', map(str, l))

#Python内建的filter()函数用于过滤序列。
print('过虑序列：')
def is_old(n):
	return  n%2 == 1

q = filter(is_old, [1, 2, 4, 5, 6, 9, 10, 15])
print(q)

#排序算法
#Python内置的sorted()函数就可以对list进行排序：
print('排序算法:')
l = [36, 5, 12, 9, 21]
print('sorted([36,5,12,9,21])=',sorted(l))
def reversed_cmp(x, y):
	if x > y:
		return -1
	if x < y:
		return 1
	return 0

#print('sorted(倒序)=', sorted(l, reversed_cmp))



