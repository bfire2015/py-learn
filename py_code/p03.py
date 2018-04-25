# -*- coding: utf-8 -*-
print('03、python中的装饰器与迭代器:')

def addTips(fun):
	def warp(*args, **kwargs):
		print('这是操作之前：')
		result = fun(*args, **kwargs)
		print('这是操作之后：');
		return result
	return warp

@addTips
def add(x, y):
	return x+y

print('add(2, 3) = ', add(2, 3))

#fortest2
def addTips(i):
    def wrap1(fun):
        def wrap(*args, **kwargs):
            print('这是操作之前')
            result = 0
            if i > 10:
                result = fun(*args, **kwargs)
            else:
                print('对不起，没有执行fun的权限')
            print('操作结束啦！')
            return result
        return wrap
    return wrap1

@addTips(11)
def add(x, y):
    return x+y

print(add(2, 3))


# 迭代器（iterator）
# 可迭代的对象：如果一个对象可以用for in 的方式遍历其内容 就是一个可迭代的对象 list tuple 字典
# 迭代器：遍历可迭代对象内容的方式
# 常见的迭代器：组合 排列 笛卡尔积  串联迭代器
# 排列 组合 笛卡尔积 串联迭代器
import itertools
x = range(1, 5)
y = list('abc')
#排列
com1 = itertools.combinations(x, 3)
#组合
com2 = itertools.permutations(x, 3)
#笛卡尔积
com3 = itertools.product(x, y)
#串联迭代器
com4 = itertools.chain(com1, com2, com3)
print('for in com4')
for i in com4:
	print(i)
