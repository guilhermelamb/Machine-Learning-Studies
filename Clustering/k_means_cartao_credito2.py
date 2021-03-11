import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('credit_card_clients.csv', header = 1)

base['TOTAL_BILL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3']\
 + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6'] 

#Using more than 2 attributes
X = base.iloc[:, [1,2,3,4,5,25]].values

scaler = StandardScaler()

X = scaler.fit_transform(X)

#Defining the number of clusters
#Within cluster sum of squares
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')

#Using 5 clusters
kmeans = KMeans(n_clusters = 5, random_state = 0)
previsoes = kmeans.fit_predict(X)

lista_clientes = np.column_stack((base, previsoes))
