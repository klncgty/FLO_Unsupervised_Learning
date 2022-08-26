import itertools

import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)


df_ = pd.read_csv(r"flo_data_20k.csv")
df = df_.copy()

df.head()
df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]') # en son kaç gün önce alışveriş yaptı
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]')

model_df = df[["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
model_df.head()

sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20),timings=False)
elbow.fit(model_df)
#plt.xlim(1,3)
#plt.ylim(0.2,0.4)
elbow.show()
elbow.k_values_
elbow.k_timers_

elbow.elbow_value_  ### optimum küme sayısı

k_means = KMeans(n_clusters = 6, random_state= 42).fit(model_df)
segments=k_means.labels_
segments

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments
final_df.head(10)
### son olarak segmentlerin istatistiksel analizi
final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","median","count"],
                                  "order_num_total_ever_offline":["mean","median","count"],
                                  "customer_value_total_ever_offline":["mean","median","count"],
                                  "customer_value_total_ever_online":["mean","median","count"],
                                  "recency":["mean","median","count"],
                                  "tenure":["mean","median","count"]})