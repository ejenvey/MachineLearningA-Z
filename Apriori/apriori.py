# Apriori

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("MachineLearningA-Z/Apriori/Market_Basket_Optimisation.csv", header=None)

#Apriori expects a list of lists
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#Training the apriori model
from apyori import apriori
rules = apriori(transactions, min_support=.0027, min_confidence =0.2, min_lift =3, min_length=2)

#Visualizing the results
results = list(rules)