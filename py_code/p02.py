# -*- coding: utf-8 -*-
#birth = int(raw_input('birth:'))
#def move(x, y, step, angle=0):
#	nx = x + step * math.cos(angle)
#	ny = y - step * math.sin(angel)
#	return nx, ny
#def fact(n):
#	if n==1:
#		return 1
#	return n * fact(n -1)
#
#fact(5)

#嵌套函数
def fun1():
	def fun2():
		print('嵌套函数fun2:')
	return fun2
fun1()()

print('函数嵌套的三层用法：')
def fun1():
	print('我是fun1的函数体语句')
	def fun2():
		print('我是fun2的函数体语句')
		def fun3():
			print('我是fun3的函数体语句')
		return fun3
	return fun2

a = fun1()
b = a()
b()
# fun1中返回 fun2的方法名
# fun1()就是调用函数 返回fun2的函数入口给变量a
# a()就是调用函数fun2 返回fun3的函数入口给变量b
# 最后调用b()

print('Python中的函数闭包')
def fun1(x):
	def fun2(y):
		#print(x + y)
		return (x + y)
	return fun2
print('fun1(10)(20) =', fun1(10)(20))
