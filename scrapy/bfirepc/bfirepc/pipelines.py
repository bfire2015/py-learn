# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from scrapy.pipelines.images import ImagesPipeline
from scrapy.exceptions import DropItem
from .items import BfirepcItem
from scrapy import Request
import hashlib
import re

class BfirepcPipeline(object):
    def process_item(self, item, spider):
        return item

class BfirepcDownloadPipeline(ImagesPipeline):
    headers = {
		'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'accept-encoding': 'gzip, deflate',
        'accept-language': 'zh-CN,zh;q=0.9,und;q=0.8,en;q=0.7',
        'cookie': 'Hm_lvt_d82cde71b7abae5cbfcb8d13c78b854c=1523436855',
        'referer': 'http://pic1.win4000.com/pic',
        'User-Agent': 'User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
        'Host': 'pic1.win4000.com',
    }
    def get_media_requests(self, item, info):
        for url in item['link']:
            #print(url)
            self.headers['referer'] = url
            yield Request(url, headers=self.headers)

    def item_completed(self, results, item, info):
        img_paths = [x['path'] for ok, x in results if ok]
        if not img_paths:
            raise DropItem("item cont no img")
        item['paths'] = img_paths
        return  item
