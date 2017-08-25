#XG Boost

#Taking code from ANN codeset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Data Preprocessing

#Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#label the country feature
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#label the gender feature
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#perform one-hot encoding
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#Avoid the dummy variable trap
X = X[:,1:]

#Splitting the dataset into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.25, random_state=0)

#Fitting XGBoost to the Training Set
from xgboost import XGBClassifier
classifier = XGBClassifier(learning_rate=0.2, gamma=0.2)
classifier.fit(X_train, y_train)

#Predict the Test set results
y_pred = classifier.predict(X_test)

#Creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Applying k-Fold Cross Validation 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y=y_train, cv = 10)
#we take the mean and standard deviation of these 10 validations
accuracies.mean()
accuracies.std()

#Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
#in the below dictionary, we create two entries which contain the parameters as keys and the parameter settings that we want to test out as the values
parameters = [{'max_depth' : [3], 'learning_rate' : [0.2], 'gamma' : [0.2], 'n_estimators' : [10, 100, 200, 1000]},
              ]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

#gives us the best accuracy, remember this is using k-fold cross validation
best_accuracy = grid_search.best_score_

#gives us a list of the best parameters selected
best_params = grid_search.best_params_