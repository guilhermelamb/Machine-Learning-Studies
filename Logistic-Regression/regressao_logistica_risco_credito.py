import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('risco_credito2.csv')

preditores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

labelencoder = LabelEncoder()
preditores[:,0] = labelencoder.fit_transform(preditores[:,0])
preditores[:,1] = labelencoder.fit_transform(preditores[:,1])
preditores[:,2] = labelencoder.fit_transform(preditores[:,2])
preditores[:,3] = labelencoder.fit_transform(preditores[:,3])



classificador = LogisticRegression()

classificador.fit(preditores, classe)

print(classificador.intercept_)
print(classificador.coef_)

resultado = classificador.predict([[0,0,1,2],[3,0,0,0]])

resultado2 = classificador.predict_proba([[0,0,1,2],[3,0,0,0]])

print(resultado)
print(resultado2)