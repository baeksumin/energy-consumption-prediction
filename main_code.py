''' import '''
import pandas as pd
import math
import re
import matplotlib.pyplot as plt 
from matplotlib import rc
import seaborn as sns
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from prophet import Prophet


''' functions '''

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

# 평가지표 (mse)
def mse(y_true, y_pred):
    return np.square(np.subtract(y_true,y_pred)).mean()

# 최적값 추적
def tuning(train, test, test_y, changepoint_prior_scale, seasonality_prior_scale, seasonality_mode, holidays_prior_scale, holidays_df):
    headers = ['changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode', 'holidays_prior_scale', 'mse']
    mse_df = pd.DataFrame([], columns = headers)

    for cps in changepoint_prior_scale:
        for sps in seasonality_prior_scale:
            for sm in seasonality_mode:
                for hps in holidays_prior_scale:
                    model = Prophet(
                        changepoint_prior_scale = cps,
                        seasonality_prior_scale = sps,
                        seasonality_mode = sm,
                        holidays_prior_scale = hps,
                        holidays = holiday
                    ).add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)\
                        .add_regressor('add1')\
                        .add_regressor('add2')\
                        .add_regressor('add3')\
                        .add_regressor('add4')\
                        .add_regressor('add5')\
                        .add_regressor('add6')\
                        .add_regressor('add7')\
                        .add_regressor('add8')
                    model.fit(train)

                    past = model.predict(test)
                    sma = mse(test_y, past['yhat'])
                    sma_list = [cps, sps, sm, hps, sma]
                    mse_df = mse_df.append(pd.Series(sma_list, index = headers), ignore_index = True)
                    print('mse_df')
                    print(mse_df)

    min_mse = mse_df[mse_df['mse'] == mse_df['mse'].min()].reset_index(drop = True)

    return mse_df, min_mse


''' main '''
pd.set_option('mode.chained_assignment',  None) # 경고 무시 설정
pd.set_option('display.max_columns', 50) # 데이터 프레임 열 출력 범위 설정

# 데이터 불러오기
data = pd.read_csv('/Users/baeksumin/apps/electricity/dataset/energy.csv', encoding = 'cp949')

from datetime import date
today = date.today()
today_ = re.sub('-', '', str(today))

# 데이터 확인
print(data.info())

# date_time 컬럼의 데이터 분할 (date, time)
data['time'] = data.date_time.str.split(' ').str[-1]
data['date'] = data.date_time.str.split(' ').str[0]
data = data[['num', 'date_time', 'date', 'time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유' ]]

# 건물별로 데이터 정렬
data_sort = data.sort_values(by=["num", "date"], ascending=[True, True], ignore_index=True)


# 상관관계 분석
data_corr = data[['전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유' ]]
rc('axes', unicode_minus=False) # 폰트 에러 문제 해결
rc('font', family='AppleGothic') # 폰트 에러 문제 해결
plt.figure(figsize=(15,10))
hm = sns.heatmap(data_corr.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
hm.get_figure().savefig("/Users/baeksumin/apps/electricity/image/heatmap.png")


# 필요한 컬럼 추가
data['불쾌지수'] = (0.81 * data.loc[:, '기온(°C)'] + 0.01 * data['습도(%)'] * (0.99 * data['기온(°C)'] - 14.3) + 46.3).round(1)
data['체감온도'] = (13.12 + 0.6215 * data['기온(°C)'] - 11.37 * data['풍속(m/s)'].apply(lambda x: math.pow(x,0.15)) + 0.3965 * data['풍속(m/s)'].apply(lambda x: math.pow(x, 0.15)) * data['기온(°C)']).round(1)
data['date_time'] = pd.to_datetime(data['date_time'])
data['weekday'] = data['date_time'].dt.weekday # 월:0, 화:1, 수:2, 목:3, 금:4, 토:5, 일:6
data_add = data[['num', 'date_time', 'date', 'weekday', 'time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '불쾌지수', '체감온도', '비전기냉방설비운영', '태양광보유']]

# 필요하지 않은 컬럼 제거
data_drop = data_add.drop(['강수량(mm)'], axis = 1) # 상관분석 결과, 전력사용량과 가장 상관도가 적은 '강수량' 컬럼 삭제

# 시각화 - 건물별 전력사용량 패턴 알기
fig = plt.figure(figsize = (15, 40))
for num in data_drop['num'].unique():
    df = data_drop[data_drop.num == num]
    df = df.groupby(['weekday', 'time'])['전력사용량(kWh)'].mean().reset_index().pivot('time', 'weekday', '전력사용량(kWh)')
    plt.subplot(12, 5, num)
    sns.heatmap(df)
    plt.title(f'building {num}')
    plt.xlabel('')
    plt.ylabel('')
    plt.yticks([])
    plt.savefig("/Users/baeksumin/apps/electricity/image/num_visualization_01.png")
# 건물별로 비슷한 전력사용량 패턴을 보이는 것이 있다는 것을 알 수 있다.


# 군집화 

data_add['time'] = data_add['time'].astype(int)
data_add['weekday'] = data_add['weekday'].astype(int)

df_list = list([])
for i in range(1, 61):
    df_list.append(data_add[data_add['num'] == i].reset_index(drop = True).iloc[:, :])

avg_list = []
for i in range(len(df_list)):
# for i in range(3):
    wd_d = [] # weekday_day
    wd_n = [] # weekday_night
    we_d = [] # weekend_day
    we_n = [] # weekend_night

    for j in range(len(df_list[i])):

        if (df_list[i]['weekday'][j] <= 4) & (9 <= df_list[i]['time'][j] <= 18): #평일 낮
            # print(df_list[i]['date_time'][j], '평일 낮')
            wd_d.append(df_list[i]['전력사용량(kWh)'][j])
        elif (df_list[i]['weekday'][j] <= 4) & ~(9 <= df_list[i]['time'][j] <= 18): #평일 밤
            # print(df_list[i]['date_time'][j], '평일 밤')
            wd_n.append(df_list[i]['전력사용량(kWh)'][j])
        elif (df_list[i]['weekday'][j] >= 5) & (9 <= df_list[i]['time'][j] <= 18): #주말 낮
            # print(df_list[i]['date_time'][j], '주말 낮')
            we_d.append(df_list[i]['전력사용량(kWh)'][j])
        elif (df_list[i]['weekday'][j] >= 5) & ~(9 <= df_list[i]['time'][j] <= 18): #주말 밤
            # print(df_list[i]['date_time'][j], '주말 밤')
            we_n.append(df_list[i]['전력사용량(kWh)'][j])

    wd_d_avg = sum(wd_d) / len(wd_d)
    wd_n_avg = sum(wd_n) / len(wd_n)
    we_d_avg = sum(we_d) / len(we_d)
    we_n_avg = sum(we_n) / len(we_n)
    avg_list = [wd_d_avg, wd_n_avg, we_d_avg, we_n_avg]
    # print(avg_list.index(max(avg_list)))
    # print(avg_list)
    # print(max(avg_list))

    df_list[i]['cluster'] = avg_list.index(max(avg_list))
    # print(df_list[i]['cluster'])
    
 
df_cluster = pd.DataFrame()
for i in range(len(df_list)):
    df_cluster = pd.concat([df_cluster, df_list[i]])


# 군집 내 군집화

con0 = df_cluster['cluster'] == 0
cluster0 = df_cluster[con0]
cluster0_group = cluster0.groupby(cluster0['num']).mean()
cluster0_group = cluster0_group[['전력사용량(kWh)']]
cluster0_group = cluster0_group.sort_values('전력사용량(kWh)')
cluster0_group = cluster0_group.reset_index()
cluster00 = cluster0_group.loc[(cluster0_group['전력사용량(kWh)'] < 2000)]['num'].tolist()
cluster01 = cluster0_group.loc[((2000 < cluster0_group['전력사용량(kWh)']) & (cluster0_group['전력사용량(kWh)'] < 5000))]['num'].tolist()
cluster02 = cluster0_group.loc[(5000 < cluster0_group['전력사용량(kWh)'])]['num'].tolist()

con2 = df_cluster['cluster'] == 2
cluster2 = df_cluster[con2]
cluster2_group = cluster2.groupby(cluster2['num']).mean()
cluster2_group = cluster2_group[['전력사용량(kWh)']]
cluster2_group = cluster2_group.sort_values('전력사용량(kWh)')
cluster2_group = cluster2_group.reset_index()
cluster20 = cluster2_group.loc[(cluster2_group['전력사용량(kWh)'] < 2000)]['num'].tolist()
cluster21 = cluster2_group.loc[((2000 < cluster2_group['전력사용량(kWh)']) & (cluster2_group['전력사용량(kWh)'] < 4000))]['num'].tolist()
cluster22 = cluster2_group.loc[(4000 < cluster2_group['전력사용량(kWh)'])]['num'].tolist()

con3 = df_cluster['cluster'] == 3
cluster3 = df_cluster[con3]
cluster3_group = cluster3.groupby(cluster3['num']).mean()
cluster3_group = cluster3_group[['전력사용량(kWh)']]
cluster3_group = cluster3_group.sort_values('전력사용량(kWh)')
cluster3_group = cluster3_group.reset_index()
cluster30 = cluster3_group.loc[(cluster3_group['전력사용량(kWh)'] < 1500)]['num'].tolist()
cluster31 = cluster3_group.loc[(1500 < cluster3_group['전력사용량(kWh)'])]['num'].tolist()

for i in range(len(df_list)):
    if df_list[i]['num'][0] in cluster00:
        df_list[i]['cluster'] = 0
    elif df_list[i]['num'][0] in cluster01:
        df_list[i]['cluster'] = 1
    elif df_list[i]['num'][0] in cluster02:
        df_list[i]['cluster'] = 2
    elif df_list[i]['num'][0] in cluster20:
        df_list[i]['cluster'] = 4
    elif df_list[i]['num'][0] in cluster21:
        df_list[i]['cluster'] = 5
    elif df_list[i]['num'][0] in cluster22:
        df_list[i]['cluster'] = 6
    elif df_list[i]['num'][0] in cluster30:
        df_list[i]['cluster'] = 7
    elif df_list[i]['num'][0] in cluster31:
        df_list[i]['cluster'] = 8
    else:
        df_list[i]['cluster'] = 3

df_recluster = pd.DataFrame()
for i in range(len(df_list)):
    df_recluster = pd.concat([df_recluster, df_list[i]])


# 군집화 시각화

fig = plt.figure(figsize = (15, 2))
for c in range(9):
    temp = df_recluster[df_recluster.cluster == c]
    temp = temp.groupby(['weekday', 'time'])['전력사용량(kWh)'].median().reset_index().pivot('time', 'weekday', '전력사용량(kWh)')

    plt.subplot(1, 10, c+1)
    sns.heatmap(temp)
    plt.title(f'cluster {c}')
    plt.xlabel('')
    plt.ylabel('')
    plt.yticks([])
# plt.savefig("/Users/baeksumin/apps/electricity/image/test3.png")


# prophet이 포맷으로 rename 
df_recluster = df_recluster[['num', 'date_time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '일조(hr)', '불쾌지수', '체감온도','비전기냉방설비운영', '태양광보유', 'cluster']]
df_recluster = df_recluster.rename(columns = {'date_time': 'ds', '전력사용량(kWh)': 'y', '기온(°C)' : 'add1', '풍속(m/s)': 'add2', '습도(%)': 'add3', '일조(hr)': 'add4', '불쾌지수': 'add5', '체감온도': 'add6','비전기냉방설비운영': 'add7', '태양광보유': 'add8', 'cluster' : 'add9'})

df_recluster = df_recluster.sort_values(by=["ds", "num"], ascending=[True, True], ignore_index=True)


train = df_recluster.loc[:112319] # 20200601~20200817
test = df_recluster.loc[112320:] # 20200818~20200824

train_df_list = list([])
test_df_list = list([])
for i in range(0, 9):
    train_df_list.append(train[train['add9'] == i].reset_index(drop = True).iloc[:, 1:])
    test_df_list.append(test[test['add9'] == i].reset_index(drop = True).iloc[:, 1:])

train_list = list([])
for i in range(0, 9):
    train_list.append(df_recluster[df_recluster['add9'] == i].reset_index(drop = True).iloc[:, 1:])

# default model

for i in range(0,9):
    model = Prophet(
        yearly_seasonality = False,
        holidays = holiday,
    ).add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
    model.fit(train_df_list[i])

    forecast = model.predict(test_df_list[i])
    model.plot(forecast)
    plt.savefig('/Users/baeksumin/apps/electricity/image/energy_future_{}_20211128.png'.format(i + 1))
    y_true = list(test_df_list[i]['y'])
    y_pred = list(forecast['yhat'])
    MSE = np.square(np.subtract(y_true,y_pred)).mean() 

    print(MSE) 

    # 군집화 후 default 모델 사용했을때, 5979302, 8402, 2477111, 847640 
    # 군집 내 군집화 후 default 모델 사용했을때, 359930, 517726, 5830013, 8402, 352623, 953409, 680317, 4685, 228111

# parameter tuning
optimum_df = pd.DataFrame([], columns = ['cluster', 'changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode', 'holidays_prior_scale', 'mse'])
for idx, val in enumerate(train_list):
    ttrain = val[val['ds'] < '2020-08-18']
    ttest = pd.DataFrame(val[val['ds'] >= '2020-08-18'].drop(['y'], axis = 1).reset_index(drop = True))
    ttest_y = val[val['ds'] >= '2020-08-18']['y'].reset_index(drop = True)

    mse_df, min_mse = tuning(ttrain, ttest, ttest_y, [0.001, 0.01, 0.1, 0.5], [0.01, 0.1, 1, 10], ['additive', 'multiplicative'], [0.01, 0.1, 1, 10], holiday)
    print('========================================== cluster {} result =========================================='.format(idx + 1))
    print(min_mse)
    print('=====================================================================================================')
    mse_df.to_csv('/Users/baeksumin/apps/electricity/dataset/mse_df/cluster_{}_{}.csv'.format(idx + 1, today_), encoding = 'UTF-8', index = False)
    num = pd.DataFrame([idx + 1], columns = ['num'])
    print('/n', '---------------------------------', num, '---------------------------------', '/n')
    num_min_mse = pd.concat([num, min_mse], axis = 1)
    optimum_df = pd.concat([optimum_df, num_min_mse], axis = 0).reset_index(drop = True)

optimum_df.to_csv('/Users/baeksumin/apps/electricity/dataset/optimum_df/{}.csv'.format(today_), encoding = 'UTF-8', index = False)


# read optimum
optimum_df = pd.read_csv('/Users/baeksumin/apps/electricity/dataset/optimum_df/20211128.csv', encoding = 'UTF-8')

# 모델 학습
answer_list = list([])
for i in tqdm(range(len(optimum_df))):
    cps = optimum_df.loc[i, 'changepoint_prior_scale']
    sps = optimum_df.loc[i, 'seasonality_prior_scale']
    sm = optimum_df.loc[i, 'seasonality_mode']
    hps = optimum_df.loc[i, 'holidays_prior_scale']
    model = Prophet(
        changepoint_prior_scale = cps,
        seasonality_prior_scale = sps,
        seasonality_mode = sm,
        holidays_prior_scale = hps,
        holidays = holiday
    ).add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)\
        .add_regressor('add1')\
        .add_regressor('add2')\
        .add_regressor('add3')\
        .add_regressor('add4')\
        .add_regressor('add5')\
        .add_regressor('add6')\
        .add_regressor('add7')\
        .add_regressor('add8')
    model.fit(train_df_list[i])

    
    # 예측
    true_y = train_list[i][train_list[i]['ds'] >= '2020-08-18'][['ds','y']].reset_index(drop = True)
    forecast = model.predict(test_df_list[i])
    answer_list = answer_list + list(forecast['yhat'])
    
    model.plot(forecast)
    # plt.plot(true_y['ds'], true_y['y'], 'r', linestyle='dashed')

    plt.savefig('/Users/baeksumin/apps/electricity/image/forecast_energy_future_{}_{}.png'.format(i + 1, today_))





