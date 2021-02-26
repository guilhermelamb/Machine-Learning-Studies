import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

base = pd.read_csv('credit_data.csv')

base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:,4].values

imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer = imputer.fit(previsores[:,1:4])
previsores[:,1:4] = imputer.transform(previsores[:,1:4])

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

classificador = GaussianNB()


previsores.shape[0]
zeros_previsores = np.zeros(shape=(previsores.shape[0],1))

kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

resultados = []
classificador = GaussianNB()
for indice_treinamento, indice_teste in kfold.split(previsores, zeros_previsores):
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoes)
    resultados.append(precisao)

resultados = np.asarray(resultados)
resultados.mean()
resultados.std()