import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('credit_data.csv')

print(base.head())

base.describe()

base.tail()

base.dtypes

base.loc[base['age']<0]


## Tratando valores inconsistentes

#para tratar esses casos onde a idade é menor do que 0

#1º deletar a coluna inteira (aqui não faria o meor sentido fazer isso, mas só 
#pra ver como é o comando)

teste_drop = base
teste_drop.drop('age', 1, inplace=True)

#2º apagar só os registros com problema (onde age <0), tb não pode ser a melhor
#ideia, pois esses registros podem conter outras infos importantes
#aqui fez uma mask para filtrar pelo valores onde age < 0

teste_filtro = base
teste_filtro.drop(teste_filtro[teste_filtro.age < 0].index, inplace=True)
teste_filtro.loc[teste_filtro['age']<0]

#3º preencher esses valores manualmente, a técnica que seria a mais correta,
#mas provavelmente a mais impraticável
#podemos preencher os valores errados com a méida de idade de todas as pessoas 
#da base

teste_preencher = base
teste_preencher.mean()
teste_preencher['age'].mean()

#Cuidado aqui, pois o valor da média está errado, está contabilizando as pessoas
#com idade negativa, devemos filtrar elas
teste_preencher['age'][teste_preencher.age > 0].mean()

#aqui o loc vai procurar os valores da coluna age que são < 0 e substituir pelo
#valor indicado (no caso a média das idades positivas)
teste_preencher.loc[teste_preencher.age < 0, 'age'] = 40.92
teste_preencher.loc[teste_preencher.age < 0]

#vamos usar este método na base de dados

base.loc[base.age < 0, 'age'] = 40.92
base.loc[base['age'] < 0]
## Tratando valores faltantes (missing values)

#Verificar valores faltantes na coluna age
pd.isnull(base['age'])

#Assim vai mostrar só os que são nulos
base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:,4].values

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(previsores[:, 0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])


scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
















