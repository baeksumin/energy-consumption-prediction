import pandas as pd

# 데이터 불러오기
data = pd.read_csv('/Users/baeksumin/apps/electricity/dataset/energy.csv', encoding = 'cp949')

# date_time 컬럼의 데이터 분할 (date, time)
data['time'] = data.date_time.str.split(' ').str[-1]
data['date'] = data.date_time.str.split(' ').str[0]
data = data[['num', 'date', 'time', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유' ]]

# 날짜별로 데이터 정렬
data_sort = data.sort_values(by=["date", "num"], ascending=[True, True], ignore_index=True)

# train, test 데이터 분할
train = data_sort.loc[:112319] # 20200601~20200817
test = data_sort.loc[112320:] # 20200818~20200824

