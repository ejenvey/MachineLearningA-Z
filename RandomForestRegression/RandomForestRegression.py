# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('MachineLearningA-Z/RandomForestRegression/Position_Salaries.csv')
#always good to make sure the X is a matrix, because most functions take it that way
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Fitting the Random Forest Regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X,y)

#Predicting a new result
y_pred = regressor.predict(6.5)

#Visualize the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color="red")
plt.plot(X_grid,regressor.predict(X_grid), color="blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position")
plt.ylabel("Salary")

