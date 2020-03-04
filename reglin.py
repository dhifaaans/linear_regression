#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:54:24 2020

@author: nadhifasofia
"""

# Nadhifa Sofia | created at March 4, 2020
# Importing the Libraries for this Linear Regression Project

import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression

# Importing the Dataset
# Dataset is scraped from https://www.the-numbers.com/movie/budgets/all/2501
data = pd.read_csv('movie2.csv')
data.head()

# Preprocessing Input Data
X = data.iloc[:,3]
Y = data.iloc[:,4]
plt.scatter(X, Y)

plt.title('Worldwide Gross vs Production Budget')
plt.xlabel('Production Budget')
plt.ylabel('Worldwide Budget')
plt.show()

# Preprocessing Data by using Selected Columns in Python
X = data.iloc[:,3]
Y = data.iloc[:,4]
dataset = pd.concat([X, Y], axis=1, join='outer')
dataset.head()

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Split Dataset into Training and Test Set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                   Y,
                                                   test_size = 1/3,
                                                   random_state = 0)

# Fitting Simple Linear Regression into Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict result of Test Set
X_test = np.nan_to_num(X_test)
Y_pred = regressor.predict(X_test)
print(Y_pred)

# Visualizing the Training Set results
plt.scatter(X_train, Y_train, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')

plt.title('Worldwide Gross vs Production Budget')
plt.xlabel('Production Budget')
plt.ylabel('Worldwide Budget')
plt.show()
