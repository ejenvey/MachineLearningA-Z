#Polynomial Regression

##Basic business problem: negotiating salary with an individual who is between manager
##and director level, essentially.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('MachineLearningA-Z/PolynomialRegression/Position_Salaries.csv')
#always good to make sure the X is a matrix, because most functions take it that way
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#in order to make the most accurate prediction possible, because we have so few records,
#we don't partition (not sure if that is kosher or not)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
##we'll transform X from a 1 column matrix to a multi-column matrix with each a power of
##itself
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly,y)

#Visualize the Linear Regression results
plt.scatter(X,y, color="red")
plt.plot(X,linear_regressor.predict(X), color="blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")

#Visualize the Polynomial Regression results
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color="red")
plt.plot(X_grid,poly_regressor.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")

#Predict a new result with the Linear Regression
linear_regressor.predict(6.5)

#Predict a new result with the Polynomial Regression 
poly_regressor.predict(poly_reg.fit_transform(6.5))
