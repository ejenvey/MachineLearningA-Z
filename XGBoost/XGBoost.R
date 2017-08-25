# XGBoost

#Read in data
dataset <- read.csv("Churn_Modelling.csv")
dataset <- dataset[4:14]

#Encode the categorical variables as factors (and converting to numeric for the NN package)

dataset$Geography = as.numeric(factor(dataset$Geography, 
                                      levels = c("Germany", "France", "Spain"),
                                      labels = c(1,2,3)))

dataset$Gender = as.numeric(factor(dataset$Gender, 
                                   levels = c("Female", "Male"),
                                   labels = c(1,2)))

#Partition the data
library(caTools)
set.seed(123)

split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Fitting XGBoost to the Training Set
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-11]), label=training_set$Exited, nrounds = 10)

#Applying k-Fold Cross Validation
library(caret)
folds = createFolds(training_set$Exited, k=10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ] #remove the current fold x
  test_fold = training_set[x,] #only the current fold x
  classifier = xgboost(data = as.matrix(training_set[-11]), label=training_set$Exited, nrounds = 10)
  y_pred <- predict(classifier, newdata = as.matrix(test_fold[-11]))
  y_pred = (y_pred > 0.5)
  cm <- table(test_fold[,11],y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})

accuracy = mean(as.numeric(cv))

#Predict the test set result
prob_pred <- h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred = as.vector(y_pred)

#Making the confusion matrix
cm <- table(test_set[,11],y_pred)

