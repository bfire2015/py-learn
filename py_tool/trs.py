#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""基于命令行的快速翻译,可直接翻译单词或句子

使用的百度翻译API

使用：

1. 直接输入文本

    >> ./trs.py "Hello World"
    >> 你好世界

    >> ./trs.py "你好世界"
    >> Hello world

2. 翻译文件

    >> echo "Hello World" > test.txt
    >> ./trs.py test.txt
    >> 你好世界

3. 推荐使用方法

    >> sudo cp ./trs.py /usr/bin/trs
    >> sudo chmod +x /usr/bin/trs

    >> trs "Hello World"
    >> 你好世界

    >> trs test.txt
    >> 你好世界

"""

# sys
import os
import sys
import urllib
import urllib.request
import json
import datetime  
import time
import hashlib


class Translation(object):

    """ 翻译 """

    # config
    BASE_URL = "http://api.fanyi.baidu.com/api/trans/vip/translate"
    API_APPID = '2015063000000001'
    API_SALT = time.ctime()
    API_KEY = "12345678"

    def __init__(self, input_args, from_="auto", to="auto"):
        self.from_ = from_
        self.to = to
        self.timeout = 10
        if len(input_args) != 2:
            print('Error: Enter a or sentence!')
            sys.exit(1)
        self.word_or_sentence = self.parse_args(input_args)
        self.translate()

    @staticmethod
    def parse_args(input_args):
        """解析命令
        """
        word_or_file = input_args[1]
        # 是否是文件
        if os.path.exists(word_or_file):
            with open(word_or_file) as fop:
                return fop.read().strip()
        return word_or_file

    def translate(self):
        """翻译 #	构建URL和参数
        """
		#	构建URL和参数
        url = self.BASE_URL
        data=self.build_url_params()
        #print(url)
        #print(data)
        try:
            response = urllib.request.urlopen(url, data=data, timeout=self.timeout)
            if response.code != 200:
                print("Error: Network error %s" % response.code)
                sys.exit(1)
            content = response.read()
        except urllib.request.HTTPError as exc:
            print("Error: %s" % str(exc))
            sys.exit(1)
        res = json.loads(content)
        self.parse_result(res)

    def build_url_params(self):
        """构建请求参数
        """
        sing = self.API_APPID + self.word_or_sentence + self.API_SALT + self.API_KEY
        sing = hashlib.md5(sing.encode(encoding='UTF-8')).hexdigest()
        params = {
            "from": self.from_,
            "to": self.to,
            "appid": self.API_APPID,
            "salt": self.API_SALT,
            "sign": sing,
            "q": self.word_or_sentence,
        }
        data = urllib.parse.urlencode(params).encode(encoding='UTF8')
        return    data

    def parse_result(self, res):
        """解析结果

        @res, dict, API返回的数据
        """
        if "error_code" in res:
            print("Error: %s %s" \
                % (res["error_msg"], res["error_code"]))
            sys.exit(1)
        results = res["trans_result"]
        dest = '\n'.join([item["dst"] for item in results])
        print(dest)


def main():
    """ main """
    Translation(sys.argv)

if __name__ == "__main__":
    main()
