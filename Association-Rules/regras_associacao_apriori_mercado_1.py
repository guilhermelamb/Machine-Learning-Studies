import pandas as pd
from apyori import apriori

#import data
base = pd.read_csv('mercado.csv', header = None)

#get the data into a list
transacoes = []

for i in range(0, 10):
    transacoes.append([str(base.values[i, j]) for j in range(0, 4)])
    
#creating rules
regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.8, 
                 min_lift = 2, min_lenght = 2)

resultados = list(regras)
print(resultados)

resultados_limpos = [list(x) for x in resultados]
resultados_limpos

resultados_formatados = []

for j in range(0, 3):
    resultados_formatados.append([list(x) for x in resultados_limpos[j][2]])
    
resultados_formatados