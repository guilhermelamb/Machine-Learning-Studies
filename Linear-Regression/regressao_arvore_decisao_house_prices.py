import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


base = pd.read_csv('house_prices.csv')

#Using multiple independent variable
X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 0)


regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

#overfitting?
score = regressor.score(X_train, y_train)


previsoes = regressor.predict(X_test)

mae = mean_absolute_error(y_test, previsoes)

regressor.score(X_test, y_test)

