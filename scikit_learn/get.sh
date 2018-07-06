#!/bin/bash
#DIR="/f/tool/py/scikit_learn/sptest"
#cd $DIR
WORKDIR=$(cd `dirname $0`;pwd)
DIR=$WORKDIR'/sptest'
cd $DIR
DATADIR=$DIR"/data"
#scrapy crawl boss -o boss20180703.json -t json
#python jsontodo.py
#python estest.py

#如果文件夹不存在，创建文件夹
if [ ! -d $DATADIR ]; then
  mkdir $DATADIR
fi

#python estest.py
#python jsontodo.py
pwd
#exit 1
boss='boss'
bossphp='bossphp'
today=`date '+%Y%m%d'`
bossfile=$DATADIR"/"$boss""$today".json"
bossphpfile=$DATADIR"/"$bossphp""$today".json"
#boss ai 爬虫
if [ ! -f $bossfile ];then
	scrapy crawl $boss -o "./data/"$boss""$today".json" -t json
else
	echo "文件已存在="$bossfile
fi

sleep 2

#boss php 爬虫
if [ ! -f $bossphpfile ];then
	scrapy crawl $bossphp -o "./data/"$bossphp""$today".json" -t json
else
	echo "文件已存在="$bossphpfile
fi

ls $DATADIR"/"$boss""$today".json"
ls $DATADIR"/"$bossphp""$today".json"

sleep 2
#转换json格式
python jsontodo.py

sleep 2
#json数据入库
python estest.py
pwd
