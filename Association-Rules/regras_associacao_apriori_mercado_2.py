import pandas as pd
from apyori import apriori

#import data
base = pd.read_csv('mercado2.csv', header = None)

#get the data into a list
transacoes = []

for i in range(0, 7501):
    transacoes.append([str(base.values[i, j]) for j in range(0, 20)])
    
#creating rules
regras = apriori(transacoes, min_support = 0.003, min_confidence = 0.2, 
                 min_lift = 3, min_lenght = 2)

#getting results
resultados = list(regras)
print(resultados)

resultados_limpos = [list(x) for x in resultados]
resultados_limpos

resultados_formatados = []

for j in range(0, 10):
    resultados_formatados.append([list(x) for x in resultados_limpos[j][2]])
    
resultados_formatados