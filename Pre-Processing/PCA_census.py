import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

base = pd.read_csv('census.csv')

previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values


label_previsores = LabelEncoder()

previsores[:,1] = label_previsores.fit_transform(previsores[:,1])
previsores[:,3] = label_previsores.fit_transform(previsores[:,3])
previsores[:,5] = label_previsores.fit_transform(previsores[:,5])
previsores[:,6] = label_previsores.fit_transform(previsores[:,6])
previsores[:,7] = label_previsores.fit_transform(previsores[:,7])
previsores[:,8] = label_previsores.fit_transform(previsores[:,8])
previsores[:,9] = label_previsores.fit_transform(previsores[:,9])
previsores[:,13] = label_previsores.fit_transform(previsores[:,13])


scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores,classe, test_size=0.15, random_state=0)


#Create PCA
pca = PCA(n_components = 6)

previsores_train = pca.fit_transform(previsores_train)
previsores_test = pca.transform(previsores_test)

componentes = pca.explained_variance_ratio_

classificador = RandomForestClassifier(n_estimators = 40, criterion='entropy', random_state=0)
classificador.fit(previsores_train,classe_train)


previsao = classificador.predict(previsores_test)


precisao = accuracy_score(classe_test,previsao)
matriz = confusion_matrix(classe_test, previsao)