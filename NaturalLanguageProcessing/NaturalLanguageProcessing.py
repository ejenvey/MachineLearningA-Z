#Natural Language Processing

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
dataset = pd.read_csv("/Users/ejenvey/MachineLearningA-Z/NaturalLanguageProcessing/Restaurant_Reviews.tsv", delimiter = "\t", quoting=3)

#Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
#remove "...", convert to lowercase, split into words
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]).lower().split()
    #create a Porter Stemmer object
    ps = PorterStemmer()
    #then, filter out all words that are in nltk's stopwords package (the set function makes it faster)
    #and, perform stemming (or standardization) to remove plurality, tense, etc. to lessen the 
    #total number of words we need to consider
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #finish the process by joining each word separately
    review = str(' '.join(review))
    corpus.append(review)

#Create bag of words model: 

    #1) build a sparse matrix of word counts
    #2) build a classification model with a target variable of positive or negative review
    #3) independent variables will be the existence of a certain word

#sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
#using max_features to reduce the words to the top 1500 in terms of count, which reduces sparsity
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

##Classification Models: Random Forest v. Naive Bayes
#Splitting the dataset into training and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.20, random_state=0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the Random Forest Classifier to the Training Set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)

#Predict the Test set results
y_pred = classifier.predict(X_test)

#Creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

accuracy_randomforest = float((cm[0,0] + cm[1,1]))/float((cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]))
precision_randomforest = float(cm[1,1])/float(cm[0,1] + cm[1,1])
f1_randomforest = 2 * precision_randomforest * accuracy_randomforest/(precision_randomforest+accuracy_randomforest)

#Naive Bayes
# Fitting the Naive Bayes Classifier to the Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predict the Test set results
y_pred = classifier.predict(X_test)

#Creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

accuracy_naivebayes = float((cm[0,0] + cm[1,1]))/float((cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]))
precision_naivebayes = float(cm[1,1])/float(cm[0,1] + cm[1,1])
f1_naivebayes = 2 * precision_randomforest * accuracy_randomforest/(precision_randomforest+accuracy_randomforest)
