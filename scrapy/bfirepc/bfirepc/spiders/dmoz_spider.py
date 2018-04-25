#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/4/11'
"""
import scrapy

from ..items import BfirepcItem

class DmozSpider(scrapy.Spider):
	name = 'dmoz'
	allowed_domains = ['dmoz.org']
	headers = {
		'User-Agent':'User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
		'Host':'www.win4000.com',
		'Referer':'http://www.win4000.com/meitu.html',
	}

	def start_requests(self):
		urls = [
#			'http://www.win4000.com/meinv146050.html',
#			'http://www.win4000.com/meinv146045.html',
			'http://www.win4000.com/meitu.html'
		]
		for url in urls:
			yield scrapy.Request(url=url, headers=self.headers, callback=self.parse)

	def parse(self, response):
#下载
#		filename = response.url.split("/")[-1]
#		with open(filename, 'wb') as f:
#			f.write(response.body)
#获取数据
		item = BfirepcItem()
		a_space = response.xpath('//div[@class="tab_box"]/*/ul[@class="clearfix"]/li/a')
		#print(a_space)
		for a in a_space:
			item['title'] = a.xpath('p/text()').extract()
			item['link'] = a.xpath('img/@data-original').extract()
			#print(item)
			yield item
