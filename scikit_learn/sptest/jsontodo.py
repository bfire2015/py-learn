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

# 今天日期
today = datetime.date.today()

# 昨天时间
yesterday = today - datetime.timedelta(days=1)

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

doc = './boss20180611.json'
newdoc = 'newboss20180611.json'
path = doc  # 数据文件路径
file = open(path, 'r', encoding='utf-8')
json_info = json.load(file)

#print(json_info)
for js_obj in json_info:
	#print(len())
	obj = js_obj
	obj['title'] = js_obj['title'][0]
	obj['money'] = js_obj['money'][0]
	obj['descs'] = js_obj['descs'][0]
	obj['company'] = js_obj['company'][0]
	obj['address'] = js_obj['wtime'][0].strip()
	obj['edu'] = ''
	#print(len(js_obj['wtime']), js_obj['wtime'])
	if len(js_obj['wtime']) == 3:
		obj['edu'] = js_obj['wtime'][2]
	ptime = js_obj['ptime'][0]
	ptime = ptime.replace('发布于', '')
	if ptime == '昨天':#today -1
		ptime = time.strftime('%m{m}%d{d}',time.localtime(int(time.mktime(time.strptime(str(yesterday), '%Y-%m-%d'))))).format(m='月', d='日')
	if ptime.find('日') == -1:#today
		ptime = time.strftime('%m{m}%d{d}',time.localtime(int(time.mktime(time.strptime(str(today), '%Y-%m-%d'))))).format(m='月', d='日')
	#print(ptime,ptime2)
	obj['ptime'] = ptime
	obj['wtime'] = js_obj['wtime'][1]

	#print(js_obj['title'][0])
	#print(obj)
#print('all the end')
#print(json_info)
#数据保存
with open(newdoc, 'w', encoding='utf-8') as f:
	# indent 超级好用，格式化保存字典，默认为None，小于0为零个空格
	# f.write(json.dumps(json_info, indent=4))
	json.dump(json_info, f, indent=4)   # 和上面的效果一样

print('all the end')
