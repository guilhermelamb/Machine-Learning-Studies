import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


base = pd.read_csv('house_prices.csv')

#Using multiple independent variable
X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 0)


regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(X_train, y_train)

#overfitting?
score = regressor.score(X_train, y_train)


previsoes = regressor.predict(X_test)

mae = mean_absolute_error(y_test, previsoes)

print(regressor.score(X_test, y_test))

rmse = np.sqrt(mean_squared_error(y_test, previsoes))