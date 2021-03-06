import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1:2].values

#scaling features
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
y_scale = scaler.fit_transform(y)

regressor = MLPRegressor()
regressor.fit(X_scale, y_scale)

score = regressor.score(X_scale, y_scale)

plt.scatter(X_scale, y_scale)
plt.plot(X_scale, regressor.predict(X_scale), color = 'red')
plt.title('Regression - Artificial Neural Networks')
plt.xlabel('Idade')
plt.ylabel('Custo R$')



