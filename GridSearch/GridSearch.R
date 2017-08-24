# Grid Search

# Using K-fold Cross Validation code as basis for this analysis

#k-Fold Cross Validation

#Read in data
dataset <- read.csv("Social_Network_Ads.csv")
dataset <- dataset[,3:5]

#Partition the data
library(caTools)
set.seed(123)

split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)


#Scaling the data
training_set[,1:2] = scale(training_set[,1:1])
test_set[,1:2] = scale(test_set[,1:2])

# Fit a Kernel SVM Classifier to the dataset
library(e1071)
classifier = svm (Purchased ~., data = training_set
                  , type= 'C-classification', kernel = 'radial')

#Predict the test set result
y_pred <- predict(classifier, newdata = test_set[-3])

#Making the confusion matrix
cm <- table(test_set[,3],y_pred)

#Applying k-Fold Cross Validation
library(caret)
folds = createFolds(training_set$Purchased, k=10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ] #remove the current fold x
  test_fold = training_set[x,] #only the current fold x
  classifier = svm (Purchased ~., data = training_fold #note how we are using the training fold to train the algorithm each iteration
                    , type= 'C-classification', kernel = 'radial')
  y_pred <- predict(classifier, newdata = test_fold[-3])
  cm <- table(test_fold[,3],y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})

accuracy = mean(as.numeric(cv))

# Building a Kernel SVM and Applying Grid Search to find the best model and the best parameters
library(caret)
training_set$Purchased <- as.factor(training_set$Purchased)
classifier = train(form = Purchased ~., data=training_set, method = 'svmRadial')
classifier$bestTune #will show the best values of hyperparameters

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Classifier (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Classifier (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
