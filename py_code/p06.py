# -*- coding: utf-8 -*-
from datetime import datetime
import time
print(datetime.now())
print(time.time())
print(time.localtime())
print(time.strftime('%Y/%m/%d %H:%M:%S', time.localtime()))
time_tuple = time.strptime('2017-12-22 11:57:00', '%Y-%m-%d %H:%M:%S')
print(time.mktime(time_tuple))
