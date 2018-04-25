#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/4/19'
"""
from elasticsearch import Elasticsearch
import time

#test
es = Elasticsearch([{'host':'192.168.56.103', 'port':9200}])
#online
#es = Elasticsearch([{'host':'120.79.82.52', 'port':9200}])
indexs = time.strftime("%Y-%m")
#indexs = '2018-04'
if es.indices.exists(index=indexs) is not True:
	es.indices.create(index=indexs)

print('create doc befor')
doc = {'t':'da','log':'b','name':'bfire','txt':'bfire test 1', 'ts':int(time.time()), 'dt': time.strftime("%Y-%m-%d %H:%M:%S")}
es.index(index=indexs, doc_type='test', refresh=True, body=doc)
print('create doc after')
_where = {
	'query':{
		'match_all':{}
	}
}
info = es.search(index=indexs, doc_type='test', body=_where)
print('search mathc_all')
print(info)
print('hit info:')
for hit in info['hits']['hits']:
	print(hit['_source'])

print('all the end')



