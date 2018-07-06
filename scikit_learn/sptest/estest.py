#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/4/19'
"""
from elasticsearch import Elasticsearch
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
#sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])

from scikit_learn.sptest import logs #引入自己写的日志类

dir = './data/'
daynum = str(datetime.date.today()).replace('-', '')
#daynum = '20180705'
fmt = 'r'
logs.p('es start', fmt)

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#test
es = Elasticsearch([{'host':'192.168.56.104', 'port':9200}])
#online
#es = Elasticsearch([{'host':'120.79.82.52', 'port':9200}])
indexs = str(time.strftime("%Y-%m"))
indexslist = [indexs, indexs + 'php']
logs.p(indexslist, fmt)
for idx in indexslist:
	logs.p(idx +' todo...', fmt)
	#idx = '2018-07php'
	if es.indices.exists(index=idx) is not True:
		logs.p(idx +' create index', fmt)
		es.indices.create(index=idx)
	
	if idx == indexs:
		doctype = 'boss'
	else:
		doctype = 'bossphp'
		
	joindir = doctype + daynum +'.json'
	newdoc = dir + 'new' + joindir
	#newdoc = './newbossphp20180703.json'
	doclock = newdoc + '1'
	if os.path.exists(doclock):
		logs.p(doclock +' is exists...', fmt)
		continue
	file = open(newdoc, 'r', encoding='utf-8')# 数据文件路径
	json_info = json.load(file)
	#doctype = 'boss'
	#doctype = 'bossphp'
	logs.p('test'+idx, fmt)
	logs.p('create doc before...', fmt)
	for i in json_info:
		es.index(index=idx, doc_type=doctype, refresh=True, body=i)
	logs.p('create doc after...', fmt)
	logs.p('create doclock before...', fmt)
	with open(doclock, 'a+', encoding='utf-8') as f:
			# indent 超级好用，格式化保存字典，默认为None，小于0为零个空格
			f.write(idx + "\n")
			f.close()
	logs.p('create doclock after...', fmt)
logs.p('es the end...', fmt)

#sys.exit(0)

#indexs = '2018-07php'
#if es.indices.exists(index=indexs) is not True:
#	es.indices.create(index=indexs)
#
#print('create doc befor')
#newdoc = './newbossphp20180703.json'
#path = newdoc  # 数据文件路径
#file = open(path, 'r', encoding='utf-8')
#json_info = json.load(file)
#doctype = 'boss'
##doctype = 'bossphp'
##print(json_info)
#for i in json_info:
#	#print(i)
#	print('create doc before')
#	es.index(index=indexs, doc_type=doctype, refresh=True, body=i)
#	print('create doc after')
#print('all the end')


#es.delete(index='indexName', doc_type='typeName', id='idValue')
#es.delete(index=indexs, doc_type='test', id='1')

#doc = {'t':'da','log':'b','name':'bfire','txt':'bfire test 1', 'ts':int(time.time()), 'dt': time.strftime("%Y-%m-%d %H:%M:%S")}
#es.index(index=indexs, doc_type='test', refresh=True, body=doc)
#print('create doc after')
#_where = {
#	'query':{
#		'match_all':{}
#	}
#}
#info = es.search(index=indexs, doc_type='test', body=_where)
#info = {'hits':{'hits':{'_source':['d']}}}
#print('search mathc_all')
#print(info)
#print('hit info:')
#for hit in info['hits']['hits']:
	#print(hit['_source'])
#	pass



