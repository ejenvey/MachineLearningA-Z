#Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('MachineLearningA-Z/SimpleLinearRegression/Salary_Data.csv')

#create matrix for independent and dependent variables separately
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Partitioning Data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.33,random_state=0)

#Train the Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
regressor.coef_

#Predicting the Test set results
#will create a vector of the test set salaries
y_pred = regressor.predict(X_test)

#Visualize training set results
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualize training set results
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Salary vs. Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

