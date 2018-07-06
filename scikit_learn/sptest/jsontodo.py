#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'json文件处理'
__author__ = 'BfireLai'
__mtime__ = '2018/5/31'
"""

import json
import time
import datetime
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
#sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])

from scikit_learn.sptest import logs #引入自己写的日志类

# 今天日期
today = datetime.date.today()

# 昨天时间
yesterday = today - datetime.timedelta(days=1)

doclist = ['boss', 'bossphp']
dir = './data/'
daynum = str(datetime.date.today()).replace('-', '')
#daynum = '20180705'
fmt = 'r'
logs.p('jsontodo start', fmt)

for doc in doclist:
	joindir = doc + daynum +'.json'
	newdoc = dir + 'new' + joindir
	path = dir + joindir #数据文件路径
	if not os.path.exists(path):
		logs.p(path +' notexists...', fmt)
		continue
	if os.path.exists(newdoc):
		logs.p(newdoc +' is exists...', fmt)
		continue
#	path = './data/bossphp20180705.json'
#	newdoc = './data/newbossphp20180705.json'
	file = open(path, 'r', encoding='utf-8')
	json_info = json.load(file)
	logs.p(path +' todo...', fmt)
	for js_obj in json_info:
		obj = js_obj
		obj['title'] = js_obj['title'][0]
		obj['money'] = js_obj['money'][0]
		obj['descs'] = js_obj['descs'][0]
		obj['company'] = js_obj['company'][0]
		obj['address'] = js_obj['wtime'][0].strip()
		obj['edu'] = ''
		if len(js_obj['wtime']) == 3:
			obj['edu'] = js_obj['wtime'][2]
		ptime = js_obj['ptime'][0]
		ptime = ptime.replace('发布于', '')
		if ptime == '昨天':#today -1
			ptime = time.strftime('%m{m}%d{d}',time.localtime(int(time.mktime(time.strptime(str(yesterday), '%Y-%m-%d'))))).format(m='月', d='日')
		if ptime.find('日') == -1:#today
			ptime = time.strftime('%m{m}%d{d}',time.localtime(int(time.mktime(time.strptime(str(today), '%Y-%m-%d'))))).format(m='月', d='日')
		obj['ptime'] = ptime
		obj['wtime'] = js_obj['wtime'][1]
		#print(js_obj['title'][0])
		#print(obj)
	logs.p(path +' todo2...', fmt)
	#数据保存
	with open(newdoc, 'w', encoding='utf-8') as f:
		# indent 超级好用，格式化保存字典，默认为None，小于0为零个空格
		# f.write(json.dumps(json_info, indent=4))
		json.dump(json_info, f, indent=4)   # 和上面的效果一样

	logs.p(path +' done...', fmt)
logs.p('jsontodo the end', fmt)


# 明天时间
#tomorrow = today + datetime.timedelta(days=1)

#acquire = today + datetime.timedelta(days=2)

# 昨天开始时间戳
#yesterday_start_time = int(time.mktime(time.strptime(str(yesterday), '%Y-%m-%d')))

# 昨天结束时间戳
#yesterday_end_time = int(time.mktime(time.strptime(str(today), '%Y-%m-%d'))) - 1

# 今天开始时间戳
#today_start_time = yesterday_end_time + 1

# 今天结束时间戳
#today_end_time = int(time.mktime(time.strptime(str(tomorrow), '%Y-%m-%d'))) - 1

# 明天开始时间戳
#tomorrow_start_time = int(time.mktime(time.strptime(str(tomorrow), '%Y-%m-%d')))

# 明天结束时间戳
#tomorrow_end_time = int(time.mktime(time.strptime(str(acquire), '%Y-%m-%d'))) - 1


#try:
#    os._exit(0)
#except:
#    print('die.')

#try:
#    sys.exit(0)
#except:
#    print('die')
#finally:
#    print('cleanup')
#
#sys.exit(0)