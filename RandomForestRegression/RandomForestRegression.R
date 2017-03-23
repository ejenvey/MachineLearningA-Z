# Random Forest Regression

#Data Preprocessing

#Read in data
dataset <- read.csv("RandomForestRegression/Position_Salaries.csv")
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
# #Scaling the data
# #training_set[,2:3] = scale(training_set[,2:3])
# #test_set[,2:3] = scale(test_set[,2:3])

#Fit Random Forest Regression Model to the dataset
library(randomForest)
regressor <- randomForest(formula=Salary~., data=dataset, ntree=1000)

#Predict new result with the Random Forest Regression Model
predict(regressor, newdata=data.frame(Level=6.5))

#Visualize the Regression Model results (and smooth the curve)
library(ggplot2)
#curve smoother
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() + geom_point(aes(x=dataset$Level , y=dataset$Salary), color="red") + 
  geom_line(aes(x=x_grid, y=predict(regressor, newdata=data.frame(Level=x_grid))), color="blue") +
  ggtitle("Regression of Salary and Level") +
  xlab("Level") +
  ylab("Salary")
