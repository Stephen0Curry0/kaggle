import numpy as np
import matplotlib
import seaborn as sns
import openpyxl
import pymysql
import sqlalchemy
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from sqlalchemy import create_engine, text, false
pd.set_option('display.unicode.east_asian_width', True)
plt.rcParams["axes.unicode_minus"]= False
plt.rcParams["font.sans-serif"]="SimHei"
from IPython.display import display
from sklearn.preprocessing import MultiLabelBinarizer
import re


#加载数据集,这里用utf-8会报错，因为cast字段（演员名）中有西欧字符
netflix_titles = pd.read_csv("netflix_titles.csv", encoding="latin1")

#专注于TV部分，drop=true表示会把原来的行索引扔掉
netflix_shows = netflix_titles[(netflix_titles.type == 'TV Show')].reset_index(drop=True)

#查看前五行数据
netflix_shows.head()

#数据清洗和预处理(快速浏览一下数据的状态，并对要做的事情做一个粗略的计划)
#查看基本信息（总行数，每列的非空值数量以及类型等），查看每个字段有几种不同值，查看每个字段的空值数量
netflix_shows.info()
netflix_shows.nunique()
netflix_shows.isna().sum(axis=0)

#接下来要做的处理：删除不必要的列（Show_id不包含任何相关信息，type只有一个唯一的值（“TV Shows”））；
#处理缺失值；把数据类型恢复正常（date_added&duration要转换类型）；整理合并列；

#删除不必要的列
netflix_shows.drop(["show_id","type"],axis=1,inplace=True)

#处理缺失值(director,cast,country,date_added,rating列有空值)
#对于director,cast,country这种明确的特征用Unknown值填充，date_added用虚拟日期1800-1-1填充；rating用已有NR（无评级）填充
netflix_shows[["director","cast","country"]]=netflix_shows[["director","cast","country"]].fillna("Unknown")
netflix_shows["date_added"]=netflix_shows["date_added"].fillna("1800-1-1")
netflix_shows["rating"]=netflix_shows["rating"].fillna("NR")

#修正数据类型：date_added要被改成datetime类型，duration要被改成数字类型
#先输出没修改的前5行内容，方便后续做对比
print("修正数据类型之前：")
print(netflix_shows[['duration', 'date_added']].head())

#将duration从object类型转变到int类型，并移除关键词“Season”(这里是按空格区分，然后取第一部分，也就是数字部分)
netflix_shows["duration"]=netflix_shows.duration.apply(lambda x:x.split(" ")[0])
netflix_shows["duration"]=netflix_shows["duration"].astype(int)

#将date_added分成年月日格式，先转成str类型
netflix_shows["date_added"]=netflix_shows["date_added"].astype(str)
#消除第一个字符是空格的异常内容
netflix_shows["date_added"]=netflix_shows["date_added"].apply(lambda x: x[1:] if x[0]==" " else x)

#获取年月日信息
tmp = pd.to_datetime(netflix_shows["date_added"],errors='coerce',format="%d-%b-%y")
netflix_shows["date_added_year"]=tmp.dt.year
netflix_shows["date_added_month"]=tmp.dt.month
netflix_shows["date_added_day"]=tmp.dt.day

#获取完之后再次填充空值，防止有未提取到的年月日信息是NaT
netflix_shows['date_added_year'] = netflix_shows['date_added_year'].fillna(1800).astype(int)
netflix_shows['date_added_month'] = netflix_shows['date_added_month'].fillna(1).astype(int)
netflix_shows['date_added_day'] = netflix_shows['date_added_day'].fillna(1).astype(int)

#将date_added转换为datetime格式
netflix_shows['date_added'] = pd.to_datetime(
    netflix_shows[['date_added_year', 'date_added_month', 'date_added_day']]
        .rename(columns={'date_added_year': 'year',
                         'date_added_month': 'month',
                         'date_added_day': 'day'})
)

#展示修改之后的
print("修正数据类型之后：")
print(netflix_shows[["duration","date_added","date_added_year","date_added_month","date_added_day"]].head())


#解开合并列(即某一列的一些单元格中有多个元素，分开来更好数据分析)，用MultiLabelBinarizer
print("使用MultiLabelBinarizer前：")
print(netflix_shows["listed_in"].head())
col="listed_in"
mlb=MultiLabelBinarizer()
netflix_shows[col] = netflix_shows[col].fillna("Unknown")
netflix_shows[col] = netflix_shows[col].apply(lambda x: x.split(", "))

one_hot = pd.DataFrame(
    mlb.fit_transform(netflix_shows[col]),
    columns=[f"{col}_{re.sub(' ', '', c)}" for c in mlb.classes_]
)

netflix_shows = netflix_shows.join(one_hot)

#删除原始列
netflix_shows.drop(columns=[col], axis=1, inplace=True)

print("使用MultiLabelBinarizer后：")
print(netflix_shows[netflix_shows.columns[netflix_shows.columns.str.startswith('listed_in')]].head())

#以上就是数据清理和预处理的全过程，接下来创建三个新特性，feature engineering

#1.星期几上架,结果是一个0-6的整数：0表示星期一,6表示星期日,可以用来分析Netflix喜欢在周几上架新剧
netflix_shows["date_added_weekday"]=pd.DatetimeIndex(netflix_shows["date_added"]).weekday

#2.首季发行年份，例如release_year为2021，duration为2，那这里结果就是2019
netflix_shows['first_release_year'] = netflix_shows.release_year - netflix_shows.duration

#3.从首季发行到 Netflix 上架的间隔年数
netflix_shows["time_first_release_to_netflix"]=netflix_shows.date_added_year - netflix_shows.first_release_year

#保存内容到csv文件中
netflix_shows.to_csv("netflix_titles_clean.csv",index=false)



#重新导入数据
netflix_shows=pd.read_csv("netflix_titles_clean.csv",encoding="latin1")

#新建画布
fig=plt.figure(figsize=(12,6),facecolor="pink")

#绘制rating的种类和数量之间的柱状图TV-Y: 2-6 years；TV-Y7: >= 7 year；TV-G: all ages；TV-PG: parental guidance
#TV-14: >= 14 years；TV-MA: mature adult
sns.countplot(x=netflix_shows["rating"], hue=netflix_shows["rating"],palette='Reds')
netflix_shows["rating"].value_counts().plot(kind="bar")

#观察图像可得结论：1.TV-MA是最多的，其次是TV-14和TV-PG，这说明大多数电视节目还是为了比较成熟的观众准备的；
#2.R这一列和TV-Y7-FV是不应该存在的列。3.最少的是TV-G，其次是TV-Y和TV-Y7，这些都是为年龄比较小的准备的电视节目

#根据结论2进一步对数据进行清理
netflix_shows["rating"]=netflix_shows["rating"].replace({"R":"TV-MA","TV-Y7-FV":"TV-Y7"})

#再重新绘制一次
sns.countplot(x=netflix_shows["rating"], hue=netflix_shows["rating"],palette='Reds')

#绘制电视剧加入Netflix的年月日和星期几信息
fig,ax=plt.subplots(2,2,figsize=(12,8))
sns.countplot(x=netflix_shows["date_added_weekday"], hue=netflix_shows["date_added_weekday"], palette='Reds', ax=ax[0][0])
sns.countplot(x=netflix_shows["date_added_day"], hue=netflix_shows["date_added_day"], palette='Reds', ax=ax[0][1])
sns.countplot(x=netflix_shows["date_added_month"], hue=netflix_shows["date_added_month"], palette='Reds', ax=ax[1][0])
sns.countplot(x=netflix_shows["date_added_year"], hue=netflix_shows["date_added_year"], palette='Reds', ax=ax[1][1])

#结论：1.周五加进来的电视节目是最多的。2.周一和周末加进来的电视节目较少。
#3.大部分电视节目都在每个月的1号或15号加入4.从加入月份来看，比较平均，但是下半年会更多一点。

#对listed_in进行可视化分析
temp = netflix_shows[netflix_shows.columns[netflix_shows.columns.str.startswith('listed_in')]].sum(axis=1)
sns.countplot(x=temp, hue=temp,palette='Reds')

#对数据进一次进行清洗："TV Shows"可以去掉
netflix_shows.drop("listed_in_TVShows", axis=1, inplace=True)

#查看首季发行到 Netflix 上架的间隔年数的可视化分析
fig=plt.figure(figsize=(12,6))
sns.histplot(netflix_shows["time_first_release_to_netflix"][netflix_shows.date_added_year!=1800],color="r",bins=50)

#duration可视化分析
sns.histplot(netflix_shows.duration, color="b")

# 调整子图布局，防止重叠
plt.tight_layout()

#显示图像
plt.show()

#保存内容
netflix_shows.to_csv("netflix_titles_clean.csv",index=false)
