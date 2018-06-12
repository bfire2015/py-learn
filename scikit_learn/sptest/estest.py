#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/4/19'
"""
from elasticsearch import Elasticsearch
import time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

#设置字符集,防止中文乱码
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#test
es = Elasticsearch([{'host':'192.168.56.104', 'port':9200}])
#online
#es = Elasticsearch([{'host':'120.79.82.52', 'port':9200}])
indexs = time.strftime("%Y-%m")
#indexs = '2018-04'
if es.indices.exists(index=indexs) is not True:
	es.indices.create(index=indexs)

print('create doc befor')
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

newdoc = './newboss20180611.json'
path = newdoc  # 数据文件路径
file = open(path, 'r', encoding='utf-8')
json_info = json.load(file)
doctype = 'boss'
#print(json_info)
for i in json_info:
	#print(i)
	print('create doc before')
	es.index(index=indexs, doc_type=doctype, refresh=True, body=i)
	print('create doc after')
print('all the end')



