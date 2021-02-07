import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC


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



classificador = SVC(C = 2, kernel = 'rbf', random_state = 1)

classificador.fit(previsores_train, classe_train)

resultado = classificador.predict(previsores_test)

accuracy = accuracy_score(classe_test, resultado)
matrix = confusion_matrix(classe_test, resultado)