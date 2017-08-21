#Data Preprocessing

#Read in data
dataset <- read.csv("Data.csv")
# dataset <- dataset[,2:3]

#Partition the data
library(caTools)
set.seed(123)

split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)


#Scaling the data
#training_set[,2:3] = scale(training_set[,2:3])
#test_set[,2:3] = scale(test_set[,2:3])

#Encoding categorical variables as factors
dataset$Country = factor(dataset$Country, 
                         levels = c("Germany", "France", "Spain"),
                         labes = c(1,2,3))