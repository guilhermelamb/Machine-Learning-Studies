import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


base = pd.read_csv('house_prices.csv')

#Using only area as independent variable
X = base.iloc[:, 5].values
y = base.iloc[:, 2].values

X = X.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

score = regressor.score(X_train, y_train)

plt.scatter(X_train, y_train)
plt.plot(X_train, regressor.predict(X_train), color = 'red')

#predicting values
previsoes = regressor.predict(X_test)


#looking the difference between real and predicted values
resultado = abs(y_test - previsoes)
resultado.mean()

mae = mean_absolute_error(y_test, previsoes)
mse = mean_squared_error(y_test, previsoes)

plt.scatter(X_test, y_test)
plt.plot(X_test, regressor.predict(X_test), color = 'red')