import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



base = pd.read_csv('credit_data.csv')

base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:,4].values

imputer = SimpleImputer()
imputer = imputer.fit(previsores[:,1:4])
previsores[:,1:4] = imputer.transform(previsores[:,1:4])

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores,classe, test_size=0.25, random_state=0)

classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)

classificador.fit(previsores_train, classe_train)

previsoes = classificador.predict(previsores_test)


precisao = accuracy_score(classe_test, previsoes)
matriz = confusion_matrix(classe_test, previsoes)

#sem rodar a parte de escalonamento (StandardScaler), obteve uma precisao de 0.836
#uma precisao menor até que o baseline classifier, ou seja, nem valeria a pena usar este algoritmo

dummy_classifier = DummyClassifier(strategy='most_frequent')
dummy_classifier.fit(previsores_train, classe_train)

previsoes_dummy = dummy_classifier.predict(previsores_test)
precisao_dummy = accuracy_score(classe_test, previsoes_dummy)
matriz_dummy = confusion_matrix(classe_test, previsoes_dummy)

#realizando o escalonamento, usando o StandardScaler, o kNN obteve uma precisao de 0.986
#um ganho de 15% na precisao





