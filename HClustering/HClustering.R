#Hierarchical clustering

#Read in dataset
dataset = read.csv("~/MachineLearningA-Z/HClustering/Mall_Customers.csv")
X <- dataset[,4:5]

#Plot Dendrogram to find optimal number of clusters
dendrogram = hclust(dist(X,method="euclidean"),method="ward.D")
plot(dendrogram, main=paste("Dendrogram"), xlab='Customers',ylab='Euclidean Distances')

#Fit the hierarchical clustering algorithm to X
hc = hclust(dist(X,method="euclidean"),method="ward.D")
y_hc = cutree(hc,5)

#Plot the clusters (2D only)
library(cluster)
clusplot(X,
         y_hc, 
         lines=0, 
         shade=TRUE, 
         color=TRUE, 
         labels=2, 
         plotchar=FALSE,
         span=TRUE, 
         main=paste('Clusters of clients'), 
         xlab="Annual Income",
         ylab="Spending Score")