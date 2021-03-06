import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler


base = pd.read_csv('house_prices.csv')

#Using multiple independent variable
X = base.iloc[:, 3:19].values
y = base.iloc[:, 2:3].values

scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
y_scale = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scale, y_scale, test_size = 0.3,
                                                    random_state = 0)


regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)


score_train = regressor.score(X_train, y_train)
score_test = regressor.score(X_test, y_test)

previsoes = regressor.predict(X_test)


y_test = scaler.inverse_transform(y_test)
previsoes = scaler.inverse_transform(previsoes)

mae = mean_absolute_error(y_test, previsoes)
rmse = np.sqrt(mean_squared_error(y_test, previsoes))