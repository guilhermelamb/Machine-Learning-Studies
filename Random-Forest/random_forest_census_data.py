import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


base = pd.read_csv('census.csv')

previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values


label_previsores = LabelEncoder()

labels = label_previsores.fit_transform(previsores[:,1])

previsores[:,1] = label_previsores.fit_transform(previsores[:,1])
previsores[:,3] = label_previsores.fit_transform(previsores[:,3])
previsores[:,5] = label_previsores.fit_transform(previsores[:,5])
previsores[:,6] = label_previsores.fit_transform(previsores[:,6])
previsores[:,7] = label_previsores.fit_transform(previsores[:,7])
previsores[:,8] = label_previsores.fit_transform(previsores[:,8])
previsores[:,9] = label_previsores.fit_transform(previsores[:,9])
previsores[:,13] = label_previsores.fit_transform(previsores[:,13])

onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])

previsores = onehotencoder.fit_transform(previsores).toarray()


label_classe = LabelEncoder()

classe = label_classe.fit_transform(classe)

scaler = StandardScaler()
previsores[:,0:1] = scaler.fit_transform(previsores[:,0:1])
previsores[:,2:3] = scaler.fit_transform(previsores[:,2:3])
previsores[:,4:5] = scaler.fit_transform(previsores[:,4:5])
previsores[:,10:11] = scaler.fit_transform(previsores[:,10:11])
previsores[:,11:12] = scaler.fit_transform(previsores[:,11:12])
previsores[:,12:13] = scaler.fit_transform(previsores[:,12:13])

previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores,classe, test_size=0.15, random_state=0)

classificador = RandomForestClassifier(n_estimators = 40, criterion='gini', random_state=0)
classificador.fit(previsores_train,classe_train)


previsao = classificador.predict(previsores_test)


precisao = accuracy_score(classe_test,previsao)
matriz = confusion_matrix(classe_test, previsao)
