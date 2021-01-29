import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export

base = pd.read_csv('risco_credito.csv')

preditores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

labelencoder = LabelEncoder()
preditores[:,0] = labelencoder.fit_transform(preditores[:,0])
preditores[:,1] = labelencoder.fit_transform(preditores[:,1])
preditores[:,2] = labelencoder.fit_transform(preditores[:,2])
preditores[:,3] = labelencoder.fit_transform(preditores[:,3])

classificador = DecisionTreeClassifier(criterion='entropy')

classificador.fit(preditores,classe)

print(classificador.feature_importances_)

for i in range(len(classificador.classes_)):
    print('Classe: {} \nGanho: {}\n'.format(classificador.classes_[i], classificador.feature_importances_[i]))


export.export_graphviz(classificador,
                       out_file = 'arvore.dot',
                       feature_names = ['historia','divida','garantias','renda'],
                       class_names = ['alto','moderado','baixo'],
                       filled=True,
                       leaves_parallel=True)


resultado = classificador.predict([[0,0,1,2],[3,0,0,0]])
print(resultado)



