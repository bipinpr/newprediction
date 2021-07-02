
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:, :-1]	
Y = dataset.iloc[:, -1:]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
new_salary_pred = regressor.predict([[12]])
print('The predicted salary of a person  is ',new_salary_pred)

import pickle
pickle.dump(regressor, open("model.pkl", "wb"))