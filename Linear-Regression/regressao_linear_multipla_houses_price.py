import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


base = pd.read_csv('house_prices.csv')

#Using multiple independent variable
X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

score = regressor.score(X_train, y_train)

#predicting values
previsoes = regressor.predict(X_test)


#looking the difference between real and predicted values
resultado = abs(y_test - previsoes)
resultado.mean()

mae = mean_absolute_error(y_test, previsoes)
mse = mean_squared_error(y_test, previsoes)

regressor.score(X_test, y_test)

print(regressor.intercept_)
print(regressor.coef_)