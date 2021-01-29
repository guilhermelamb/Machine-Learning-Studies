import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

base = pd.read_csv('risco_credito.csv')

preditores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

labelencoder = LabelEncoder()
preditores[:,0] = labelencoder.fit_transform(preditores[:,0])
preditores[:,1] = labelencoder.fit_transform(preditores[:,1])
preditores[:,2] = labelencoder.fit_transform(preditores[:,2])
preditores[:,3] = labelencoder.fit_transform(preditores[:,3])



classificador = GaussianNB()

classificador.fit(preditores, classe)

resultado = classificador.predict([[0,0,1,2], [3,0,0,0]])

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)
