#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://{}:{}@{}/{}?charset={}".
                       format('root', 'root', '127.0.0.1:3306', 'findwork','utf8'))
score = pd.read_sql("select * from score", engine)
student = pd.read_sql("select * from student", engine)
score.最终 = 0
student.必修得分 = 0
student.必修否 = 0
print("数据集score的条数为：\n", score)
print("数据集student的条数为：\n", student)
#第一题
print("---------第一题--------")
a = (score.groupby('课程',as_index=False).count())
a = a[['课程','平时']]
a = a.rename(columns={'平时':'总人数'})
print(a)
print("---------第一题结束--------")
#第二题
print("---------第二题--------")
score.最终 = score.平时*0.3+score.期末*0.7
for r1 in range(student.shape[0]):
    for r2 in range(score.shape[0]):
        if (student.iloc[r1]['必修课'] == score.iloc[r2]['课程']) and (student.iloc[r1]['id'] == score.iloc[r2]['id']):
            #下面不能写为data1.iloc[r1]['b'],这样写数据不会更新
            student.iloc[r1,student.columns == '必修得分'] = score.iloc[r2,score.columns == '最终']
print("n2数据集score为：\n", score)
print("n2数据集student为：\n", student)
print("---------第二题结束--------")

#第三题
print("---------第三题--------")
for r1 in range(student.shape[0]):
    w1 = student.loc[r1]['选课']
    w2 = student.loc[r1]['必修课']
    if(w2 in w1):
        student.loc[r1, student.columns == '必修否']=1
ns = student.loc[student['必修否']==0]
print(ns)
print("---------第三题结束--------")

#第四题
print("---------第四题--------")
score3 = score.loc[score['课程']=='体育']
max = score3['最终'].max()
mean = score3['最终'].mean()
df = pd.DataFrame({
    '最高分':[max],
    '平均分':[mean],
})
print(df)
print("---------第四题结束--------")

#第五题
print("---------第五题--------")
score5 = score.loc[score['课程']=='体育']
max = score5['最终'].max()
score5 = score5.loc[score5["最终"]<max]

score5.sort_values(by="最终",axis=0,ascending=False,inplace=True)
score5 = score5.reset_index(drop = True)
print(score5)
id = score5.loc[0]['id']
student5 = student.loc[student['id']==id]
print(student5)
print("---------第五题结束--------")