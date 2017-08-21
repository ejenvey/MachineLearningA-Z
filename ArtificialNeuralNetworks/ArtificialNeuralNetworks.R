# Artificial Neural Networks

#Read in data
dataset <- read.csv("Churn_Modelling.csv")
dataset <- dataset[,4:14]

#Encode the categorical variables as factors (and converting to numeric for the NN package)

dataset$Geography = as.numeric(factor(dataset$Geography, 
                         levels = c("Germany", "France", "Spain"),
                         labels = c(1,2,3)))

dataset$Gender = as.numeric(factor(dataset$Gender, 
                           levels = c("Female", "Male"),
                           labels = c(1,2)))

#Partition the data
library(caTools)
set.seed(123)

split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Feature Scaling
training_set[,-11] = scale(training_set[,-11])
test_set[,-11] = scale(test_set[,-11])

# Connect to H2O instance, then fit ANN to the dataset
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y= 'Exited',
                              training_frame = as.h2o(training_set),
                              activation = "Rectifier",
                              hidden = c(6,6),
                              epochs = 100,
                              train_samples_per_iteration = -2)

#Predict the test set result
prob_pred <- h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred = as.vector(y_pred)

#Making the confusion matrix
cm <- table(test_set[,11],y_pred)

#Disconnect h2o
h2o.shutdown()