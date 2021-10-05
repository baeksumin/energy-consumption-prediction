import pandas as pd
import math
import matplotlib.pyplot as plt 
from matplotlib import rc
import seaborn as sns
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans

pd.set_option('mode.chained_assignment',  None) # 경고 무시

# 데이터 불러오기
data = pd.read_csv('/Users/baeksumin/apps/electricity/dataset/energy.csv', encoding = 'cp949')

# 데이터 확인
data.info() 

# date_time 컬럼의 데이터 분할 (date, time)
data['time'] = data.date_time.str.split(' ').str[-1]
data['date'] = data.date_time.str.split(' ').str[0]
data = data[['num', 'date_time', 'date', 'time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유' ]]

# 날짜별로 데이터 정렬
data_sort = data.sort_values(by=["date", "num"], ascending=[True, True], ignore_index=True)

# 상관관계 분석
data_corr = data[['전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유' ]]
rc('axes', unicode_minus=False) # 폰트 에러 문제 해결
rc('font', family='AppleGothic') # 폰트 에러 문제 해결
plt.figure(figsize=(15,10))
hm = sns.heatmap(data_corr.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
hm.get_figure().savefig("/Users/baeksumin/apps/electricity/image/heatmap.png")

# train, test 데이터 분할
train = data_sort.loc[:112319] # 20200601~20200817
test = data_sort.loc[112320:] # 20200818~20200824

# 필요한 컬럼 추가
train['불쾌지수'] = (0.81 * train.loc[:, '기온(°C)'] + 0.01 * train['습도(%)'] * (0.99 * train['기온(°C)'] - 14.3) + 46.3).round(1)
train['체감온도'] = (13.12 + 0.6215 * train['기온(°C)'] - 11.37 * train['풍속(m/s)'].apply(lambda x: math.pow(x,0.15)) + 0.3965 * train['풍속(m/s)'].apply(lambda x: math.pow(x, 0.15)) * train['기온(°C)']).round(1)
train['date_time'] = pd.to_datetime(train['date_time'])
train['weekday'] = train['date_time'].dt.weekday
train_add = train[['num', 'date_time', 'date', 'weekday', 'time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유' ]]
# print(train_add)

# 필요하지 않은 컬럼 제거
# 상관분석 결과, 전력사용량과 가장 상관도가 적은 '강수량' 컬럼 삭제
train_drop = train_add.drop(['강수량(mm)'], axis = 1)
print(train_drop)

# 군집화
# 건물을 기준으로 하는 data frame 생성
by_weekday = train_drop.groupby(['num','weekday'])['전력사용량(kWh)'].median().reset_index().pivot('num','weekday','전력사용량(kWh)').reset_index()
by_time = train_drop.groupby(['num','time'])['전력사용량(kWh)'].median().reset_index().pivot('num','time','전력사용량(kWh)').reset_index().drop('num', axis = 1)
df = pd.concat([by_weekday, by_time], axis= 1)
columns = ['num'] + ['day'+str(i) for i in range(7)] + ['time'+str(i) for i in range(24)]
df.columns = columns
# print(df)

# '전력사용량'이 아닌 '요일과 시간대에 따른 전력 사용량의 경향성'에 따라서만 군집화 할 것이므로, 특수한 scaling이 필요함
# standard scaling
for i in range(len(df)):
    # 요일 별 전력 중앙값에 대해 scaling
    df.iloc[i,1:8] = (df.iloc[i,1:8] - df.iloc[i,1:8].mean())/df.iloc[i,1:8].std()
    # 시간대별 전력 중앙값에 대해 scaling
    df.iloc[i,8:] = (df.iloc[i,8:] - df.iloc[i,8:].mean())/df.iloc[i,8:].std()

# visualization
fig = plt.figure(figsize = (10,3))
# 요일별
for i in range(len(df)):
    plt.plot(df.iloc[i, 1:8], alpha = 0.5, linewidth = 0.5)
# plt.savefig("/Users/baeksumin/apps/electricity/image/clustering_week.png")
# 시간별
fig = plt.figure(figsize = (20,3))
for i in range(len(df)):
    plt.plot(df.iloc[i, 8:], alpha = 0.5, linewidth = 0.5)
# plt.savefig("/Users/baeksumin/apps/electricity/image/clustering_time.png")

# k-means clustering
# elbow method를 통해 군집의 개수 결정
def change_n_clusters(n_clusters, data):
    sum_of_squared_distance = []
    for n_cluster in n_clusters:
        kmeans = KMeans(n_clusters=n_cluster)
        kmeans.fit(data)
        sum_of_squared_distance.append(kmeans.inertia_)
        
    plt.figure(1 , figsize = (8, 5))
    plt.plot(n_clusters , sum_of_squared_distance , 'o')
    plt.plot(n_clusters , sum_of_squared_distance , '-' , alpha = 0.5)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    # plt.savefig("/Users/baeksumin/apps/electricity/image/count_of_clusters.png")

change_n_clusters([2,3,4,5,6,7,8,9,10,11], df.iloc[:,1:])
# 그래프 확인 결과 최적 군집 수는 4로 결정

