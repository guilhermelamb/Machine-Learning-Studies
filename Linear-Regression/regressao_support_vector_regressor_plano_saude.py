import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1:2].values

#linear kernel
regressor_linear = SVR(kernel = 'linear')
regressor_linear.fit(X, y)

score_linear = regressor_linear.score(X, y)

plt.scatter(X, y)
plt.plot(X, regressor_linear.predict(X), color = 'red')
plt.title('Support Vector Regressor - Linear')
plt.xlabel('Idade')
plt.ylabel('Custo R$')

#polynomial kernel
regressor_poly = SVR(kernel = 'poly', degree = 3)
regressor_poly.fit(X, y)

score_poly = regressor_poly.score(X, y)

plt.scatter(X, y)
plt.plot(X, regressor_poly.predict(X), color = 'red')
plt.title('Support Vector Regressor - Polynomial')
plt.xlabel('Idade')
plt.ylabel('Custo R$')

#rbf kernel
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
y_scale = scaler.fit_transform(y)

regressor_rbf = SVR(kernel = 'rbf')
regressor_rbf.fit(X_scale, y_scale)

score_rbf = regressor_rbf.score(X_scale, y_scale)

plt.scatter(X_scale, y_scale)
plt.plot(X_scale, regressor_rbf.predict(X_scale), color = 'red')
plt.title('Support Vector Regressor - RBF')
plt.xlabel('Idade')
plt.ylabel('Custo R$')




