#!/usr/bin/env python
# coding: utf-8

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

#Fetaure Selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
#Modelling Algoritm
from sklearn.cluster import KMeans, MiniBatchKMeans

import folium
from folium.plugins import FastMarkerCluster, Fullscreen, MiniMap, HeatMap, HeatMapWithTime, LocateControl
import requests
import json

from wordcloud import WordCloud

# # 4. Modelado
# 
# Segmentaremos los clientes utilizando k-means con análisis RFM (Recency, Frequency, Monetary value) que es uno de los métodos de segmentación de clientes más sencillos de implantar, y al mismo tiempo uno de los que mejores resultados aportan a corto plazo.
# 
# - Recency: ¿Cuándo fue la última vez que un Cliente me compró algo?
# - Frecuency: ¿Cuántas veces me ha comprado un Cliente en el periodo de análisis?
# - Monetary value: ¿Cuál ha sido el valor monetario agregado de un Cliente en dicho periodo?
# 

# ## 4.1 Recency, Recency y Monetary value
# 
# Deberemos crear estas variables para cada cliente antes de realizar la segmentacion.

df = pd.DataFrame(all_data['customer_id'].unique())
df.columns = ['customer_id']

# ### 4.1.2 Recency
# Para calcular recency, vamos utilizar la delivered order_purchase_timestamp

recency_df = all_data.groupby('customer_id').order_purchase_timestamp.max().reset_index()
recency_df.columns = ['customer_id', 'max_purchase_date']
recency_df['recency'] = (recency_df['max_purchase_date'].max() - recency_df['max_purchase_date']).dt.days
df = pd.merge(df, recency_df[['customer_id','recency', 'max_purchase_date']], on='customer_id')
df.head()

# ### 4.1.2 Frequency
# Para encontrar frequency, es necesario encontrar el numero total de pedidos para cada cliente.

frequency_df = all_data.groupby('customer_id').order_item_id.count().reset_index()
frequency_df.columns = ['customer_id','frequency']
df = pd.merge(df, frequency_df, on='customer_id')
df.head()

# ### 4.1.3 Monetary Value
# Para encontrar el monetary value sumamos todos los pedidos realizados por cada cliente.

monetary_df = all_data['payment_value'].groupby(all_data['customer_id']).sum().reset_index()
monetary_df.columns = ['customer_id','monetary_value']
df = pd.merge(df, monetary_df, on='customer_id')
df.head()

# ## 4.2 RFM Clusters K-means
# Aplicaremos el algoritmo K-Means simultaneamente para todas las variables RFM. Pero antes, realizaremos el tratamiento de los outliers através del método IQR.

df_rm = df[['recency', 'monetary_value']]
df_rm.head()

colnames = ['recency', 'frequency', 'monetary_value']

for col in colnames:
    fig, ax = plt.subplots(figsize=(8,5))
    sns.distplot(df[col])
    ax.set_title('Distribución de %s' % col)
    plt.show()

plt.figure(figsize=(8,5))
df['recency'].plot.hist()  

plt.figure(figsize=(8,5))
df['frequency'].plot.hist()

plt.figure(figsize=(8,5))
df['monetary_value'].plot.hist()

# ### 4.2.1 Tratamiento de outliers
# Para encontrar el monetary value sumamos todos los pedidos realizados por cada cliente.

# Funcion para el tratamiento de los outliers
def outlier_treatment(df):
    numerical_columns = df.select_dtypes(include=['int32','int64', 'float64']).columns
    for column in numerical_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3-Q1
        df.loc[ df[column] > (1.5*IQR), column ] = ( 1.5*IQR )
        df.loc[ df[column] < (-1.5*IQR), column] = ( -1.5*IQR )

outlier_treatment(df_rm)
df_rm.describe()

# ### 4.2.2 Normalizacion
df_rm_normal = df_rm.copy()

scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(df_rm_normal)
df_rm_normal = scaler.transform(df_rm_normal)

df_rm_normal = pd.DataFrame(df_rm_normal, index=df_rm.index)
df_rm_normal.columns=['recency', 'monetary_value']

# Check result after standardising
df_rm_normal.describe()

# ### 4.2.3 Clusterizacion

X=df_rm_normal.copy()
Nc = range(1, 11)
kmeans = [None] * 10
for i in Nc:
    kmeans[i-1] = KMeans(n_clusters=i)

# score = [None] * 10
# for i in range(len(kmeans)):
#     score[i] = (kmeans[i].fit(X).inertia_) 

# plt.figure(figsize=(15,9))
# plt.plot(Nc,score)
# plt.xlabel('Cantidad de clusters')
# plt.ylabel('Score')
# plt.title('Elbow Curve (Curva codo)')
# plt.xticks(Nc)
# plt.show()

# El gráfico nos muestra que el número ideal de clusters es 5.

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# KMeans
X=df_rm_normal.copy()
kmeans = KMeans(n_clusters=5)
model = kmeans.fit(X)

X['Labels_KMeans'] = model.labels_

clusters = X[['Labels_KMeans']]
clusters.columns = ['cluster']

rm_cluster5 = df_rm.copy()
rm_cluster5 = pd.concat([rm_cluster5, clusters], axis=1, sort = False)

# sns.relplot(x="monetary_value", y="recency", hue='cluster',
#             sizes=(50, 500), alpha=.3,palette=sns.color_palette('hls', 5),
#             height=5, data=rm_cluster5)


rm_cluster5.groupby('cluster')['recency','monetary_value'].mean()
rm_cluster5['cluster'].value_counts()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MiniBatchKMeans
X=df_rm_normal.copy()

miniBatchKMean = MiniBatchKMeans(n_clusters=5, random_state=0, batch_size=100)

model = miniBatchKMean.fit(X)

X['Labels_KMeans'] = model.labels_

clusters = X[['Labels_KMeans']]
clusters.columns = ['cluster']

rm_cluster5 = df_rm.copy()
rm_cluster5 = pd.concat([rm_cluster5, clusters], axis=1, sort = False)

# sns.relplot(x="monetary_value", y="recency", hue='cluster',
#             sizes=(50, 500), alpha=.3,palette=sns.color_palette('hls', 5),
#             height=5, data=rm_cluster5)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Incremental k-means 

from river import cluster
from river import metrics

X = df_rm_normal.iloc[0:50]

incremental_kmeans = cluster.KMeans(n_clusters=5, halflife=0.4, sigma=3, seed=0)
metric_ssw = metrics.cluster.SSW()
metric_cohesion = metrics.cluster.Cohesion()
metric_separation = metrics.cluster.Separation()
metric_ssb = metrics.cluster.SSB()
metric_bic = metrics.cluster.BIC()
metric_silhouette = metrics.cluster.Silhouette()
metric_xieBeni = metrics.cluster.XieBeni()

for row in X.to_dict('records'):
    incremental_kmeans = incremental_kmeans.learn_one(row)
    prediction = incremental_kmeans.predict_one(row)
   
    metric_ssw = metric_ssw.update(row, prediction, incremental_kmeans.centers)
    metric_cohesion = metric_cohesion.update(row, prediction, incremental_kmeans.centers)
    metric_separation = metric_separation.update(row, prediction, incremental_kmeans.centers)
    metric_ssb = metric_ssb.update(row, prediction, incremental_kmeans.centers)
    metric_bic = metric_bic.update(row, prediction, incremental_kmeans.centers)
    metric_silhouette = metric_silhouette.update(row, prediction, incremental_kmeans.centers)
    metric_xieBeni = metric_xieBeni.update(row, prediction, incremental_kmeans.centers)
    
prediction = incremental_kmeans.predict_one({'recency': 1.1912261428503317, 'monetary_value': 0.5009208384397379})    
print('Predicion cluster nro:',prediction)
print()
print('Metrica', metric_ssw)
print('Mas grande es mejor: ', metric_ssw.bigger_is_better)
print()
print('Metrica', metric_cohesion)
print('Mas grande es mejor:', metric_cohesion.bigger_is_better)
print()
print('Metrica', metric_separation)
print('Mas grande es mejor:', metric_separation.bigger_is_better)
print()
print('Metrica', metric_ssb)
print('Mas grande es mejor:', metric_ssb.bigger_is_better)
print()
print('Metrica', metric_bic)
print('Mas grande es mejor:', metric_bic.bigger_is_better)
print()
print('Metrica', metric_silhouette)
print('Mas grande es mejor:', metric_silhouette.bigger_is_better)
print()
print('Metrica', metric_xieBeni)
print('Mas grande es mejor:', metric_xieBeni.bigger_is_better)

centers_df = pd.DataFrame(incremental_kmeans.centers).transpose()
centers_df

sns.relplot(x="monetary_value", y="recency", alpha=.9, height=8, data=centers_df)

X = df_rm_normal.iloc[100:7500]

for row in X.to_dict('records'):
    incremental_kmeans = incremental_kmeans.learn_one(row)
    prediction = incremental_kmeans.predict_one(row)
    metric_ssw = metric_ssw.update(row, prediction, incremental_kmeans.centers)
    metric_cohesion = metric_cohesion.update(row, prediction, incremental_kmeans.centers)
    metric_separation = metric_separation.update(row, prediction, incremental_kmeans.centers)
    metric_ssb = metric_ssb.update(row, prediction, incremental_kmeans.centers)
    metric_bic = metric_bic.update(row, prediction, incremental_kmeans.centers)
    metric_silhouette = metric_silhouette.update(row, prediction, incremental_kmeans.centers)
    metric_xieBeni = metric_xieBeni.update(row, prediction, incremental_kmeans.centers)
    
prediction = incremental_kmeans.predict_one({'recency': 1.1912261428503317, 'monetary_value': 0.5009208384397379})    
print('Predicion cluster nro:',prediction)
print()
print('Metrica', metric_ssw)
print('Mas grande es mejor: ', metric_ssw.bigger_is_better)
print()
print('Metrica', metric_cohesion)
print('Mas grande es mejor:', metric_cohesion.bigger_is_better)
print()
print('Metrica', metric_separation)
print('Mas grande es mejor:', metric_separation.bigger_is_better)
print()
print('Metrica', metric_ssb)
print('Mas grande es mejor:', metric_ssb.bigger_is_better)
print()
print('Metrica', metric_bic)
print('Mas grande es mejor:', metric_bic.bigger_is_better)
print()
print('Metrica', metric_silhouette)
print('Mas grande es mejor:', metric_silhouette.bigger_is_better)
print()
print('Metrica', metric_xieBeni)
print('Mas grande es mejor:', metric_xieBeni.bigger_is_better)


centers_df = pd.DataFrame(incremental_kmeans.centers).transpose()
centers_df

sns.relplot(x="monetary_value", y="recency", alpha=.9, height=8, data=centers_df)

df_cluster = df.copy()
outlier_treatment(df_cluster)
cluster_label = rm_cluster5['cluster']
df_cluster = pd.concat([df_cluster, cluster_label], axis=1, sort = False)

plt.figure(figsize = (15,9))
colors = ['#118ab2', '#8ac926','#ff6392','#f77f00','#ffca3a']
ax = sns.relplot(x="monetary_value", y="recency", hue='cluster',
            sizes=(50, 500), alpha=.6,palette=colors,
            height=5, data=df_cluster)
plt.show()

df_cluster['cluster'].value_counts(normalize=True)*100

colors = ['#118ab2', '#8ac926','#ff6392','#f77f00','#ffca3a']

plt.figure(figsize=(10,7))
ax = df_cluster['cluster'].value_counts().sort_index().plot(kind='bar',width=0.9,color=colors, edgecolor='white')
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1))
    
plt.xlabel('Clusters')
plt.ylabel('Cantidad de clientes')
plt.title('Clientes por Cluster')
plt.xticks(fontweight='bold', rotation='horizontal')

plt.show()

all_data_cluster = all_data.merge(df_cluster, on='customer_id', how='inner')

pay = all_data_cluster.groupby('cluster')['payment_type'].value_counts()
pay = pay.unstack()
pay.reset_index()

clusters = (['0','1','2','3','4'])
boleto = np.array(pay['boleto'].unique())
credit = np.array(pay['credit_card'].unique())
debit = np.array(pay['debit_card'].unique())
voucher = np.array(pay['voucher'].unique())


total = boleto+credit+debit+voucher
proportion_voucher = np.true_divide(voucher,total)*100
proportion_debit = np.true_divide(debit,total)*100
proportion_boleto = np.true_divide(boleto,total)*100
proportion_credit = np.true_divide(credit,total)*100


colors = ['#a3acff', '#b5ffb9','#ff6392','#FFD670','#E9FF70']

# The position of the bars on the x-axis
r = range(len(clusters))

barWidth = 0.9
#plot bars
plt.figure(figsize=(10,7))
ax1 = plt.bar(r, proportion_voucher, bottom=proportion_debit+proportion_boleto+proportion_credit, color=colors[4], edgecolor='white', width=barWidth, label="voucher")
ax3 = plt.bar(r, proportion_debit, bottom = proportion_boleto+proportion_credit, color=colors[2], edgecolor='white', width=barWidth, label='debit')
ax4 = plt.bar(r, proportion_boleto, bottom = proportion_credit, color=colors[1], edgecolor='white', width=barWidth, label='boleto')
ax5 = plt.bar(r, proportion_credit , color=colors[0], edgecolor='white', width=barWidth, label='credit')


plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fancybox=True)
plt.xticks(r, clusters, fontweight='bold')
plt.xlabel('Clusters')
plt.ylabel('%')
plt.title('Medio de Pago', pad=25)
plt.show()

region = all_data_cluster.groupby('cluster')['customer_regiao'].value_counts().unstack().reset_index()

clusters = (['0','1','2','3','4'])
centro = np.array(region['Centro-Oeste'].unique())
nordeste = np.array(region['Nordeste'].unique())
norte = np.array(region['Norte'].unique())
sudeste = np.array(region['Sudeste'].unique())
sul = np.array(region['Sul'].unique())


total = centro+nordeste+norte+sudeste+sul
proportion_centro = np.true_divide(centro,total)*100
proportion_nordeste = np.true_divide(nordeste,total)*100
proportion_norte = np.true_divide(norte,total)*100
proportion_sudeste = np.true_divide(sudeste,total)*100
proportion_sul = np.true_divide(sul,total)*100

colors = ['#2b9348', '#ffff3f','#aacc00','#0081a7','#219ebc']

# The position of the bars on the x-axis
r = range(len(clusters))

barWidth = 0.9
#plot bars
plt.figure(figsize=(10,7))
ax1 = plt.bar(r, proportion_centro, bottom=proportion_nordeste+proportion_norte+proportion_sudeste+proportion_sul, color=colors[0], edgecolor='white', width=barWidth, label="Centro-Oeste")
ax2 = plt.bar(r, proportion_nordeste, bottom=proportion_norte+proportion_sudeste+proportion_sul, color=colors[1], edgecolor='white', width=barWidth, label='Nordeste')
ax3 = plt.bar(r, proportion_norte, bottom = proportion_sudeste+proportion_sul, color=colors[2], edgecolor='white', width=barWidth, label='Norte')
ax3 = plt.bar(r, proportion_sudeste,  bottom = proportion_sul, color=colors[3], edgecolor='white', width=barWidth, label='Sudeste')
ax4 = plt.bar(r, proportion_sul, color=colors[4], edgecolor='white', width=barWidth, label='Sul')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fancybox=True)
plt.xticks(r, clusters, fontweight='bold')
plt.xlabel('Clusters')
plt.ylabel('%')
plt.title('Region', pad=25)
plt.show()
