import pandas as pd
import numpy as np

# 数据分析&绘图
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import warnings

warnings.filterwarnings("ignore")

# 科学计算
from scipy.stats import skew, kurtosis
import pylab as py

# 时间
import time
import datetime
from datetime import datetime
from datetime import date
import calendar

data = pd.read_csv('data.csv')
data.info()
# 查看缺失数据
print(data.isnull().sum()[data.isnull().sum() != 0])
# 缺失值处理
# company 缺失太多，删除
# country、children和agent缺失比较少，用字段内的众数填充
# country和children用字段内的众数填充 agent缺失值用0填充，代表没有指定任何机构
data_new = data.copy(deep=True)
data_new.drop("company", axis=1, inplace=True)
data_new["agent"].fillna(0, inplace=True)
data_new["children"].fillna(data_new["children"].mode()[0], inplace=True)
data_new["country"].fillna(data_new["country"].mode()[0], inplace=True)
print(data_new.isnull().sum()[data_new.isnull().sum() != 0])
print(data_new.head(10))
print(data_new[['children', 'agent', 'country']])
data_new.info()
print(data_new.isnull().sum()[data_new.isnull().sum() != 0])

# 处理一下异常值：成人+小孩+婴儿=0的情况都需要删掉
data_new["children"] = data_new["children"].astype(int)
data_new["agent"] = data_new["agent"].astype(int)  # 转换数据类型
zero_guests = list(data_new["adults"] + data_new["children"] + data_new["babies"] == 0)
data_new.drop(data_new.index[zero_guests], inplace=True)
# meal字段映射处理
data_new["meal"].replace(["Undefined", "BB", "FB", "HB", "SC"],
                         ["No Meal", "Breakfast", "Full Board", "Half Board", "No Meal"], inplace=True)
# 数据去重
data_new.drop_duplicates(inplace=True)
data_new.to_csv('data_new.csv')
print(data.shape)
print(data_new.shape)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 相关性分析
# 相关系数矩阵
corr_matrix = round(data_new.corr(), 3)
"Correlation Matrix: "
corr_matrix.to_csv('correlation.csv')
print(corr_matrix)
# 相关系数
cancel_corr = data_new.corr()["is_canceled"]
print(cancel_corr.abs().sort_values(ascending=False)[1:])
plt.rcParams['figure.figsize'] = (30, 30)
sns.heatmap(data_new.corr(), annot=True, cmap="YlGnBu", linewidths=5)
plt.suptitle('Correlation Between Variables', fontweight='heavy', x=0.03, y=0.98, ha="left", fontsize='18',
             fontfamily='sans-serif', color="black")
plt.show()

# 计算各个特征与is_canceled相关系数
cancel_corr = data_new.corr()["is_canceled"]
print(cancel_corr.abs().sort_values(ascending=False))
