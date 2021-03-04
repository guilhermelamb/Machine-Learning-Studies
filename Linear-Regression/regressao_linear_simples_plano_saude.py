import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from yellowbrick.regressor import ResidualsPlot

base = pd.read_csv('plano_saude.csv')

X = base.iloc[:, 0].values
y = base.iloc[:, 1].values

correlacao = np.corrcoef(X, y)

#reshaping data
X = X.reshape(-1,1)

#create regressor and fit
regressor = LinearRegression()
regressor.fit(X, y)


#b0
print(regressor.intercept_)

#b1
print(regressor.coef_)

#printing chart
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title('Regressão Linear Simples')
plt.xlabel('Idade')
plt.ylabel('Custo R$');


#printing score
score = regressor.score(X, y)

#printing residuals chart
visualizador = ResidualsPlot(regressor)
visualizador.fit(X, y)
visualizador.poof();