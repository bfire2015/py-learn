#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/7/5'
"""
import time
import datetime
import codecs #防止编码问题
import os
import sys
import json

def p(desc, format='r'):
	logtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
	if type(desc)==list:
		str = ''
		for i in desc:
			str += ' ' + json.dumps(i)
		logstr = '[' + logtime + '] ' + str
	elif type(desc)==dict:
		str = ''
		for i in desc:
			str += ' ' + i
		logstr = '[' + logtime + '] ' + str
	else:
		logstr = '[' + logtime + '] ' + desc
		
	if format == 'r':
		print(logstr)
	if format == 'w':
		logdir = './logs'
		if not os.path.exists(logdir):
			os.makedirs(logdir)
		file = logdir + '/' + str(datetime.date.today()).replace('-', '') + '.log'
		with open(file, 'a+', encoding='utf-8') as f:
			# indent 超级好用，格式化保存字典，默认为None，小于0为零个空格
			f.write(logstr + "\n")
			f.close()
	if format == 1:
		print(logstr)
		sys.exit(0)



#传入的参数为path和code，path表示txt文件的绝对或相对路径，code表示该txt的编码，一般为utf-8无bom，两个参数的数据类型都为str。
def rtxt(path, code):
	with codecs.open(path, 'r', encoding=code)as f:
		txt_lines = f.readlines()
	return txt_lines

#传入参数为path、content和code，path和code和上述相同，content即为写入的内容，数据类型为字符串。
def wtxt(path, content, code):
	with codecs.open(path, 'a', encoding=code)as f:
		f.write(content)
	return path+' is ok!'