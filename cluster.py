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
# elbow method를 통해 군집의 개수 결정 elbow가 달라질 수 있음
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

