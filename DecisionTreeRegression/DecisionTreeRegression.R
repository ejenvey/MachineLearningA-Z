# Decision Tree Regression

#Data Preprocessing

#Read in data
dataset <- read.csv("DecisionTreeRegression/Position_Salaries.csv")
 dataset <- dataset[,2:3]

# #Partition the data
# library(caTools)
# set.seed(123)
# 
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# 
# training_set = subset(dataset, split==TRUE)
# test_set = subset(dataset, split==FALSE)


#Fit Decision Tree Regression Model to the dataset
#Create regressor here
library(rpart)
regressor <- rpart(formula=Salary~., data=dataset, control = rpart.control(minsplit=1))

#Predict new result with the Decision Tree Regression Model
predict(regressor, newdata=data.frame(Level=6.5))

#Below, we use the smoother.  Basically, this is a non-continuous model, and because
#it is noncontinuous, we need to represent the breaks between the predictions with vertical
#lines since the prediction is EITHER one value or another based on the values of the 
#independent variable(s)

#Visualize the Decision Tree Regression Model results (and smooth the curve)
library(ggplot2)
#curve smoother
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)

ggplot() + geom_point(aes(x=dataset$Level , y=dataset$Salary), color="red") + 
  geom_line(aes(x=x_grid, y=predict(regressor, newdata=data.frame(Level=x_grid))), color="blue") +
  ggtitle("Regression of Salary and Level") +
  xlab("Level") +
  ylab("Salary")