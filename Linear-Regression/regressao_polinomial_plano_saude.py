import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:,1].values

#Simple linear regression
regressor_linear = LinearRegression()
regressor_linear.fit(X, y)
score_linear = regressor_linear.score(X, y)

plt.scatter(X, y)
plt.plot(X, regressor_linear.predict(X), color = 'red')
plt.title('Regressão Linear Simples')
plt.xlabel('Idade')
plt.ylabel('Custo R$')

#Polynomial regression
poly = PolynomialFeatures(degree = 3)
X_poly = poly.fit_transform(X)

regressor_poly = LinearRegression()
regressor_poly.fit(X_poly, y)
score_poly = regressor_poly.score(X_poly, y)

plt.scatter(X, y)
plt.plot(X, regressor_poly.predict(X_poly), color = 'red')
plt.title('Regressão Linear Polinomial e Simples')
plt.xlabel('Idade')
plt.ylabel('Custo R$')