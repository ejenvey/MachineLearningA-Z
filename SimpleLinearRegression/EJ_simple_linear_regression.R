#Simple Linear Regression

library(ggplot2)

#Read in data
dataset <- read.csv("SimpleLinearRegression/Salary_Data.csv")
# dataset <- dataset[,2:3]

#Partition the data
library(caTools)
set.seed(123)

split = sample.split(dataset$Salary, SplitRatio = 2/3)

training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Train the model
regressor <- lm(Salary~YearsExperience,data=dataset)

#Predicting Test Set Results
y_pred <- predict(regressor,newdata=test_set)

#Visualize Training set results
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),col='red') + 
  geom_line(aes(x = training_set$YearsExperience, y = 
                  predict(regressor, newdata = training_set)),col='blue') + 
  ggtitle('Salary v Experience (Training Set)') + 
  xlab('Years of Experience') + ylab('Salary')

#Visualize Test set results
ggplot() + geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary)
                      ,col='red') + 
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor, newdata = training_set)) ,col='blue') + 
  ggtitle('Salary v Experience (Test Set)') + xlab('Years of Experience') + ylab('Salary')

