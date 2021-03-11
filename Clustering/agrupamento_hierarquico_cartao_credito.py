import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


base = pd.read_csv('credit_card_clients.csv', header = 1)

base['TOTAL_BILL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3']\
 + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6'] 

#Using more than 2 attributes
X = base.iloc[:, [1,25]].values

scaler = StandardScaler()

X = scaler.fit_transform(X)

dendrograma = dendrogram(linkage(X, method = 'ward'))

hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
previsoes = hc.fit_predict(X)

plt.scatter(X[previsoes == 0, 0], X[previsoes == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[previsoes == 1, 0], X[previsoes == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[previsoes == 2, 0], X[previsoes == 2, 1], s = 100, c = 'blue', label = 'Cluster 3')
plt.xlabel('Limite (R$)')
plt.ylabel('Gastos (R$)')
plt.legend()