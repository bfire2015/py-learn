# -*- coding: utf-8 -*-
print('04、python中的匿名函数与Lambda表达式:')

print('i^2 = ',[i**2 for i in range(8)])

print('i%2 = ', [i for i in range(9) if i%2 == 0])

a = [[1, 2, 3], [44, 55, 66], [77, 88, 99]]
print([j for i in a for j in i])

from functools import reduce

lambda1 = lambda x: x**2
lambda2 = lambda x, y: x+y
lambda3 = lambda x: x%2 == 0

#map reduce filter
print('[0,8)->i^2 = ', list(map(lambda1, range(8))))
print('[0,8)->x+y = ', reduce(lambda2, range(8)))
print('[0,8)->x%2 = ', list(filter(lambda3, range(8))))

# 练习：计算5!+4!+3!+2!+1!的和
# 要求：使用我们刚刚讲的lambda和map reduce filter
# 5!= 5*4*3*2*1

la1 = lambda x,y :x*y
print('[1,6)->x*y = ',reduce(la1, range(1,6)))

la2 = lambda n: reduce(la1, range(1, n + 1))
print('[1,6)->n = ',list(map(la2, range(1,6))))

la3 = lambda a, b: a+b
print('[1,6)->a+b = ',reduce(la3, list(map(la2, range(1,6)))))
