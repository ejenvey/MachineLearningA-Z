# Polynomial Regression

#Data Preprocessing

#Read in data
dataset <- read.csv("PolynomialRegression/Position_Salaries.csv")
dataset <- dataset[,2:3]

#Fit Linear Regression to the dataset
lin_reg <- lm(Salary~., data=dataset)

#Fit Polynomial Regression to the dataset
dataset$Level2 <- dataset$Level^2
dataset$Level3 <- dataset$Level^3
dataset$Level4 <- dataset$Level^4
poly_reg <- lm(Salary~., data=dataset)

#Visualize the Linear Regression results
library(ggplot2)
ggplot() + geom_point(aes(x=dataset$Level , y=dataset$Salary), color="red") + 
  geom_line(aes(x=dataset$Level , y=predict(lin_reg, newdata=dataset)), color="blue") +
ggtitle("Linear Regression of Salary and Level") +
xlab("Level") +
ylab("Salary")

#Visualize the Polynomial Regression results
library(ggplot2)
ggplot() + geom_point(aes(x=dataset$Level , y=dataset$Salary), color="red") + 
  geom_line(aes(x=dataset$Level , y=predict(poly_reg, newdata=dataset)), color="blue") +
  ggtitle("Linear Regression of Salary and Level") +
  xlab("Level") +
  ylab("Salary")

#Predict new result with Linear Regression
predict(lin_reg, newdata=data.frame(Level=6.5))

#Predict new result with Polynomial Regression
predict(poly_reg, newdata=data.frame(Level=6.5, Level2=6.5^2, Level3=6.5^3, Level4=6.5^4))