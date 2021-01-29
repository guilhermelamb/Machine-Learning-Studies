import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


base = pd.read_csv('census.csv')

previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values


label_previsores = LabelEncoder()

#labels = label_previsores.fit_transform(previsores[:,1])

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
previsores = scaler.fit_transform(previsores)

previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores,classe, test_size=0.15, random_state=0)



