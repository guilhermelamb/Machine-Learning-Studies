import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


base = pd.read_csv('house_prices.csv')

#Using multiple independent variable
X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 0)

poly = PolynomialFeatures(degree = 2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train_poly, y_train)

score = regressor.score(X_train_poly, y_train)

previsoes = regressor.predict(X_test_poly)

mae = mean_absolute_error(y_test, previsoes)