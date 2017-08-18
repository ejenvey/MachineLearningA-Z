# Eclat
library(arules)

#Data Preprocessing

dataset <- read.csv("Eclat/Market_Basket_Optimisation.csv", header=F)
##the below function will read the data in a way that we can do our apriori algorithm, as this is a 
##sparse matrix.
##the rm.duplicates function will remove some duplicates in the data that would prevent us from doing apriori
##the output of rm.duplicates will show in the console that, in this case, there were 5 transactions with 1
##duplicate
dataset <- read.transactions("~/MachineLearningA-Z/Apriori/Market_Basket_Optimisation.csv", sep=','
                             , rm.duplicates = TRUE)
##the below command will give you many different facts...including the density, meaning the percentage of 
##nonzero records
summary(dataset)

itemFrequencyPlot(dataset,topN=10)

#Training Eclat on the dataset

##in the case of this example, we want to look only at products that are purchased, at 
##minimum, 3 times a day...so 3*7/7500 will give you the support at 0.003

rules <- eclat(dataset, parameter = list(support = 0.004, minlen=3))

#visualizing the results
inspect(sort(rules, by='support')[1:10])
