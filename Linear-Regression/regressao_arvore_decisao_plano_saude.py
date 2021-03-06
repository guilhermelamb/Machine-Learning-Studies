import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

#Tree regressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

score = regressor.score(X, y)

plt.scatter(X, y)
plt. plot(X, regressor.predict(X), color = 'red')
plt.title('Tree Regressor')
plt.xlabel('Idade')
plt.ylabel('Custo R$')

#To see how the regressor really looks like
X_test = np.arange(min(X), max(X), 0.1)
X_test = X_test.reshape(-1, 1)
plt.scatter(X, y)
plt. plot(X_test, regressor.predict(X_test), color = 'red')
plt.title('Tree Regressor')
plt.xlabel('Idade')
plt.ylabel('Custo R$')