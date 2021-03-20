import pandas as pd
from pyod.models.knn import KNN


base = pd.read_csv('credit_data.csv')
base = base.dropna()

#Creating detector
detector = KNN()
detector.fit(base.iloc[:,1:4])

#Identifying labels
previsoes = detector.labels_

#Confidence
confiaca_previsoes = detector.decision_scores_

#Identifying the outliers
outliers_index = []

for i in range(len(previsoes)):
    if previsoes[i] == 1:
        outliers_index.append(i)

outliers = base.iloc[outliers_index, :]
        
