# -*- coding: utf-8 -*-
import pymysql
print('05、python中操作mysql数据库CRUD(增、删、改、查)')
# python操作mysql数据库的三种方式：1、pymysql  2、mysqldb 3、sqlalchemy


# 1、数据库的连接
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='', db='py', charset='utf8')
print(conn)

# 2、创建操作的游标
cursor = conn.cursor()

# 3、设置字符编码以及自动提交
cursor.execute('set names utf8')
cursor.execute('set autocommit=1')
conn.commit()

# 4、编写sql语句 crud
#注意表里 id 为主键并且要自增
#sql = "insert into py_user(name, pwd) values('bfire666', '6667')"
# sql = 'delete from py_user where id={0}'.format(2)
#sql = "update py_user set pwd='333' where name='bfire666'"
sql = 'select * from py_user'
print(sql)

# 5、执行sql并且得到结果集

cursor.execute(sql)

# 得到结果集三种方式：fetchone()  fetchemany(n)  fetchall()

result = cursor.fetchall()
# result = cursor.fetchone()
#result = cursor.fetchmany(2)
print('这是结果集：',result)

# 6、关闭游标和连接
cursor.close()
conn.close()
