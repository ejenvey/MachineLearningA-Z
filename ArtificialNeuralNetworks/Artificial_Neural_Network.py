# Artificial Neural Network

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


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Step 2: Build the Artificial Neural Network

# Import Keras library and other packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#We can define a Deep Learning model either by defining the sequence of layers or defining a graph
#Below I'll do a sequence of layer

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer (using rule of thumb of averaging the 
#number input and output layers for the amount of hidden layer neurons)
classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu', input_dim=11))

#Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu'))

#Add the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation = 'sigmoid'))

#Compiling the ANN (applying Stochastic Gradient Descent and specifying loss function)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#Predict the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

accuracy = float((cm[0,0] + cm[1,1])) / 2500.0