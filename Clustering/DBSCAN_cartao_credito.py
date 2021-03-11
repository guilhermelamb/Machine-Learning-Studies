import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


base = pd.read_csv('credit_card_clients.csv', header = 1)

base['TOTAL_BILL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3']\
 + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6'] 

#Using more than 2 attributes
X = base.iloc[:, [1,25]].values

scaler = StandardScaler()

X = scaler.fit_transform(X)

dbscan = DBSCAN(eps = 0.37, min_samples = 4)

previsoes = dbscan.fit_predict(X)

unicos, quantidade = np.unique(previsoes, return_counts = True)

plt.scatter(X[previsoes == 0, 0], X[previsoes == 0, 1], s = 100, c = 'red')
plt.scatter(X[previsoes == 1, 0], X[previsoes == 1, 1], s = 100, c = 'green')
plt.scatter(X[previsoes == 2, 0], X[previsoes == 2, 1], s = 100, c = 'blue')