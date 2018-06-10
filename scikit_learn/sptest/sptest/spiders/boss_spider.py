#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'BfireLai'
__mtime__ = '2018/05/30'
"""
import scrapy

from ..items import SptestItem

class BossSpider(scrapy.Spider):
	name = 'boss'
	allowed_domains = ['boss.com']
	headers = {
		'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
		'Host':'www.zhipin.com',
		'Referer':'https://www.zhipin.com/job_detail/?query=ai&scity=101280600&industry=&position=',
	}

	def start_requests(self):
		urls = [
			#'https://www.zhipin.com/job_detail/?query=ai&scity=101280600&industry=&position=',
			'https://www.zhipin.com/c101280600/h_101280600/?query=ai&ka=recommend-job-list'
		]
		for url in urls:
			yield scrapy.Request(url=url, headers=self.headers, callback=self.parse)

	def parse(self, response):
		#print(response)
#下载
#		filename = response.url.split("/")[-1]
#		with open(filename, 'wb') as f:
#			f.write(response.body)
#获取数据
		item = SptestItem()
		a_space = response.xpath('//div[@class="job-primary"]')
		#print(a_space)
		for a in a_space:
			item['title'] = a.xpath('div[@class="info-primary"]/*/a/div[@class="job-title"]/text()').extract()
			item['money'] = a.xpath('div[@class="info-primary"]/*/a/span[@class="red"]/text()').extract()
			item['descs'] = a.xpath('div[@class="info-primary"]/*/a/span/text()').extract()
			item['wtime'] = a.xpath('div[@class="info-primary"]/p/text()').extract()
			item['company'] = a.xpath('div[@class="info-company"]/*/h3/a/text()').extract()
			item['ptime'] = a.xpath('div[@class="info-publis"]/p/text()').extract()
			print(item['descs'])
			yield item
