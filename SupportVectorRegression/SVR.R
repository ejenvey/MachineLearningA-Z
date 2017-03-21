# SVR Regression

#Data Preprocessing

#Read in data
dataset <- read.csv("PolynomialRegression/Position_Salaries.csv")
dataset <- dataset[,2:3]

# #Partition the data
# library(caTools)
# set.seed(123)
# 
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# 
# training_set = subset(dataset, split==TRUE)
# test_set = subset(dataset, split==FALSE)
# 
# 

#Scaling the data
# dataset = scale(dataset)

#Fit the SVR to the dataset
#Create regressor here

library(e1071)
regressor <- svm(formula=Salary~.,data=dataset, type='eps-regression')

#Predict new result with the Regression Model
predict(regressor, newdata=data.frame(Level=6.5))

#Visualize the Regression Model results (and smooth the curve)
library(ggplot2)
#curve smoother
# x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

ggplot() + geom_point(aes(x=dataset$Level , y=dataset$Salary), color="red") + 
  geom_line(aes(x=dataset$Level, y=predict(regressor, newdata=dataset), color="blue") +
  ggtitle("Regression of Salary and Level") +
  xlab("Level") +
  ylab("Salary")