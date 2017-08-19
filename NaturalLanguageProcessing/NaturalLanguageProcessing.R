#Natural Language Processing

##Read dataset
dataset_original = read.delim("Restaurant_Reviews.tsv", quote='', stringsAsFactors = FALSE)

##Clean the text
library(tm)
library(SnowballC)
#First, we'll read Reviews into a corpus, then lowercase them
corpus=VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus,content_transformer(tolower))
#Then, we remove the numbers
corpus = tm_map(corpus,removeNumbers)
#Next, remove any punctuation
corpus = tm_map(corpus,removePunctuation)
#Remove nonrelevant words (pronouns, articles, etc.)
corpus = tm_map(corpus, removeWords,stopwords())
#Perform stemming (or standardization) by keeping only the root words
corpus = tm_map(corpus,stemDocument)
#Remove extra whitespace
corpus = tm_map(corpus,stripWhitespace)

##Build a sparse matrix of features (bag of words model) for Machine Learning
dtm = DocumentTermMatrix(corpus)
#Remove the .1% least frequent words (because they will not aid our ML model)
dtm = removeSparseTerms(dtm, 0.999)

##Build Random Forest Model
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

#Encode the target feature as a factor
dataset$Liked <- factor(dataset$Liked,levels=c(0,1))

#Split the data
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Fit a Random Forest Classifier to the dataset
library(randomForest)
classifier <- randomForest(x=training_set[-692],y=training_set$Liked,ntree=10)

#Predict the test set result
y_pred <- predict(classifier, newdata = test_set[-692])

#Making the confusion matrix
cm <- table(test_set[,692],y_pred)