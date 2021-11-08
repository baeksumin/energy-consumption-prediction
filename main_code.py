import pandas as pd
import math
import matplotlib.pyplot as plt 
from matplotlib import rc
import seaborn as sns
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import requests
from bs4 import BeautifulSoup
from prophet import Prophet

pd.set_option('mode.chained_assignment',  None) # 경고 무시 설정
pd.set_option('display.max_columns', 50) # 데이터 프레임 열 출력 범위 설정

# 데이터 불러오기
data = pd.read_csv('/Users/baeksumin/apps/electricity/dataset/energy.csv', encoding = 'cp949')

# 데이터 확인
# data.info() 

# date_time 컬럼의 데이터 분할 (date, time)
data['time'] = data.date_time.str.split(' ').str[-1]
data['date'] = data.date_time.str.split(' ').str[0]
data = data[['num', 'date_time', 'date', 'time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유' ]]
# print(data)

# 건물별로 데이터 정렬
data_sort = data.sort_values(by=["num", "date"], ascending=[True, True], ignore_index=True)
# print(data_sort.tail(50))

# # 상관관계 분석
# data_corr = data[['전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유' ]]
# rc('axes', unicode_minus=False) # 폰트 에러 문제 해결
# rc('font', family='AppleGothic') # 폰트 에러 문제 해결
# plt.figure(figsize=(15,10))
# hm = sns.heatmap(data_corr.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
# # hm.get_figure().savefig("/Users/baeksumin/apps/electricity/image/heatmap.png")

# # train, test 데이터 분할
# train = data_sort.loc[:112319] # 20200601~20200817
# test = data_sort.loc[112320:] # 20200818~20200824

# 필요한 컬럼 추가
data['불쾌지수'] = (0.81 * data.loc[:, '기온(°C)'] + 0.01 * data['습도(%)'] * (0.99 * data['기온(°C)'] - 14.3) + 46.3).round(1)
data['체감온도'] = (13.12 + 0.6215 * data['기온(°C)'] - 11.37 * data['풍속(m/s)'].apply(lambda x: math.pow(x,0.15)) + 0.3965 * data['풍속(m/s)'].apply(lambda x: math.pow(x, 0.15)) * data['기온(°C)']).round(1)
data['date_time'] = pd.to_datetime(data['date_time'])
data['weekday'] = data['date_time'].dt.weekday
data_add = data[['num', 'date_time', 'date', 'weekday', 'time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '불쾌지수', '체감온도', '비전기냉방설비운영', '태양광보유']]
# print(data_add)

# 필요하지 않은 컬럼 제거
# 상관분석 결과, 전력사용량과 가장 상관도가 적은 '강수량' 컬럼 삭제
data_drop = data_add.drop(['강수량(mm)'], axis = 1)
# print(data_drop)

# 시각화 - 건물별 전력사용량 패턴 알기
# fig = plt.figure(figsize = (15, 40))
# for num in train_drop['num'].unique():
#     df = train_drop[train_drop.num == num]
#     df = df.groupby(['weekday', 'time'])['전력사용량(kWh)'].mean().reset_index().pivot('weekday', 'time', '전력사용량(kWh)')
#     plt.subplot(12, 5, num)
#     sns.heatmap(df)
#     plt.title(f'building {num}')
#     plt.xlabel('')
#     plt.ylabel('')
#     plt.yticks([])
#     plt.savefig("/Users/baeksumin/apps/electricity/image/num_visualization.png")
# 건물별로 비슷한 전력사용량 패턴을 보이는 것이 있다는 것을 알 수 있다.


# 군집화
# 건물을 기준으로 하는 data frame 생성
by_weekday = data_drop.groupby(['num','weekday'])['전력사용량(kWh)'].median().reset_index().pivot('num','weekday','전력사용량(kWh)').reset_index()
by_time = data_drop.groupby(['num','time'])['전력사용량(kWh)'].median().reset_index().pivot('num','time','전력사용량(kWh)').reset_index().drop('num', axis = 1)
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



kmeans = KMeans(n_clusters=4, random_state = 2)
km_cluster = kmeans.fit_predict(df.iloc[:,1:])
df_clust = df.copy()
df_clust['km_cluster'] = km_cluster
df_clust['km_cluster'] = df_clust['km_cluster'].map({0:1, 1:3, 2:2, 3:0})
# print(df_clust)

data_drop = data_drop.merge(df_clust[['num','km_cluster']], on = 'num', how = 'left')

# visualizing result of kmeans clustering
# n_c = len(np.unique(df_clust.km_cluster)) 


# fig = plt.figure(figsize = (20, 4))
# for c in range(4):
#     temp = data_drop[data_drop.km_cluster == c]
#     temp = temp.groupby(['weekday', 'time'])['전력사용량(kWh)'].median().reset_index().pivot('weekday', 'time', '전력사용량(kWh)')
#     plt.subplot(1, 5, c+1)
#     sns.heatmap(temp)
#     plt.title(f'cluster {c}')
#     plt.xlabel('')
#     plt.ylabel('')
#     plt.yticks([])
# plt.savefig("/Users/baeksumin/apps/electricity/image/test1.png")


train_ = data_drop.merge(df_clust[['num','km_cluster']], on = 'num', how = 'left')
# print(train_)


# prophet이 포맷으로 rename 
# train_['datetime'] = pd.to_datetime(train_['date'].str.cat(train_['time'], sep='-'))
# print(train_)
# data = data[['num', 'date_time', 'date', 'time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유' ]]
# print(data)

last_data = data_drop[['num', 'date_time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '일조(hr)', '불쾌지수', '체감온도','비전기냉방설비운영', '태양광보유', 'km_cluster']]
last_data = last_data.rename(columns = {'date_time': 'ds', '전력사용량(kWh)': 'y', '기온(°C)' : 'add1', '풍속(m/s)': 'add2', '습도(%)': 'add3', '일조(hr)': 'add4', '불쾌지수': 'add5', '체감온도': 'add6','비전기냉방설비운영': 'add7', '태양광보유': 'add8', 'km_cluster' : 'add9'})
# print(last_data.head(50))

# 함수화해보자..
df0 = last_data[last_data.add9 == 0]
df0_sort = df0.sort_values(by=["ds", "num"], ascending=[True, True], ignore_index=True)
df0_drop = df0_sort.drop(['num'], axis = 1)
df0_train = df0_drop.loc[:69263] # 20200601~20200817
df0_test = df0_drop.loc[69264:] # 20200818~20200824

df1 = last_data[last_data.add9 == 1]
df1_sort = df1.sort_values(by=["ds", "num"], ascending=[True, True], ignore_index=True)
df1_drop = df1_sort.drop(['num'], axis = 1)
df1_train = df1_drop.loc[:24335] # 20200601~20200817
df1_test = df1_drop.loc[24336:] # 20200818~20200824

df2 = last_data[last_data.add9 == 2]
df2_sort = df2.sort_values(by=["ds", "num"], ascending=[True, True], ignore_index=True)
df2_drop = df2_sort.drop(['num'], axis = 1)
df2_train = df2_drop.loc[:11231] # 20200601~20200817
df2_test = df2_drop.loc[11232:] # 20200818~20200824

df3 = last_data[last_data.add9 == 3]
df3_sort = df3.sort_values(by=["ds", "num"], ascending=[True, True], ignore_index=True)
df3_drop = df3_sort.drop(['num'], axis = 1)
df3_train = df3_drop.loc[:7487] # 20200601~20200817
df3_test = df3_drop.loc[7488:] # 20200818~20200824

# 군집별로 데이터프레임을 분리하였다 !! ------------------------------------------------------------------------


# 휴일 데이터 가져오기 (공공데이터포털 API사용)
def print_whichday(year, month, day) :
    r = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    aday = dt.date(year, month, day)
    bday = aday.weekday()
    return r[bday]

def get_request_query(url, operation, params, serviceKey):
    import urllib.parse as urlparse
    params = urlparse.urlencode(params)
    request_query = url + '/' + operation + '?' + params + '&' + 'serviceKey' + '=' + serviceKey
    return request_query

year = 2020
mykey = "VygvqzZz%2FxRZ%2Bp3i119xUZJ1i2EY%2FIrsCPR0Hgtdggi6ha%2FiL4F7oKwutUm26UkjD188qyIp8WZk70a1bGqdwg%3D%3D"

date_name = []
loc_date = []
for month in range(6,9):

    month = '0' + str(month)
    
    url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService'
    operation = 'getRestDeInfo'
    params = {'solYear':year, 'solMonth':month}

    request_query = get_request_query(url, operation, params, mykey)
    get_data = requests.get(request_query)    

    if True == get_data.ok:
        soup = BeautifulSoup(get_data.content, 'html.parser')        
        
        item = soup.findAll('item')
        #print(item);

        for i in item:
            day = int(i.locdate.string[-2:])
            weekname = print_whichday(int(year), int(month), day)
            locdate = str(i.locdate.string)
            datename = str(i.datename.string)
            loc_date.append(locdate)
            date_name.append(datename)
# print(loc_date)
# print(date_name)            
# 20200606, 20200815, 20200817 공휴일 확인

# prophet이 원하는 형태로 공휴일 데이터 가공
def holidays_to_df():
    holidays = pd.DataFrame({
        'holiday' : date_name,
        'ds' : loc_date,
        'lower_window' : 0,
        'upper_window' : 0
    })
    return holidays

holiday = holidays_to_df()


# default model

# model = Prophet(
#     yearly_seasonality = False,
#     holidays = holiday,
# ).add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
# model.fit(df0_train)

# past = model.predict(df0_test)
# y_true = list(df0_test['y'])
# y_pred = list(past['yhat'])
# MSE = np.square(np.subtract(y_true,y_pred)).mean() 
# print(MSE) # default 모델 사용했을때 5115003.44164709

# df0, df1, df2, df3 전부 다 학습 가능하게.. for문 돌려서 해야돼


