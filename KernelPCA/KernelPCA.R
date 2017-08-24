#Kernel PCA

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
training_set[,1:2] = scale(training_set[,1:2])
test_set[,1:2] = scale(test_set[,1:2])

#Applying Kernel PCA
library(kernlab)
kpca = kpca(~., data = training_set[-3], kernel = "rbfdot",features = 2) #note that because PCA is unsupervised, we leave out the response variable in the training_data
training_set_pca = as.data.frame(predict(kpca, training_set)) #note that we don't have the dependent variable here, but the ordering of observations is the same, so we can simply add them back in
training_set_pca$Purchased = training_set$Purchased

test_set_pca = as.data.frame(predict(kpca, test_set)) #note that we don't have the dependent variable here, but the ordering of observations is the same, so we can simply add them back in
test_set_pca$Purchased = test_set$Purchased

#Fit Logistic Regression Model to the dataset
classifier <- glm(formula = Purchased ~ ., 
                  family="binomial", data=training_set_pca)

#Predict the test set result
prob_pred <- predict(classifier, type="response", newdata=test_set_pca[-3])
y_pred <- ifelse(prob_pred > 0.5, 1, 0)

#Making the confusion matrix
cm <- table(test_set_pca[,3],y_pred)

# Visualising the Training set results
library(ElemStatLearn)
set = training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'PC1', ylab = 'PC2',
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
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
