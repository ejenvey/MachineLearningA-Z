#Multiple Linear Regression

#Read in the data
dataset <- read.csv("MultipleLinearRegression/50_startups.csv")

# Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('California', 'New York', 'Florida'),
                         labels = c(1, 2, 3))

#Partition the data
library(caTools)
set.seed(123)

split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Fitting Multiple Linear Regression to the Training Set
regressor <- lm(formula = Profit ~ .,data=training_set)

#Predicting the test set results
y_pred <- predict(regressor,newdata=test_set)

#Optimize the model using Backward Elimination
regressor_opt <- lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State
                    ,data=training_set)

#Remove all of State (because both dummy variables are VERY insignificant)
regressor_opt <- lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend
                    ,data=training_set)
summary(regressor_opt)

#Remove Administration
regressor_opt <- lm(formula = Profit ~ R.D.Spend + Marketing.Spend
                    ,data=training_set)
summary(regressor_opt)

#Remove Marketing.Spend
regressor_opt <- lm(formula = Profit ~ R.D.Spend
                    ,data=training_set)
summary(regressor_opt)