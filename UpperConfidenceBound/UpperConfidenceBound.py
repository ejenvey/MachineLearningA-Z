#Upper Confidence Bound

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#Importing the dataset
#This dataset basically simulates an experiment where you are sending 10000 users 10 different ads, 
#and it tells us which ads the person clicks on
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implement the UCB Algorithm (without using a library)
#number of users
N = 10000
#number of ads
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0,N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if (numbers_of_selections[i]) > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            average_reward = np.float64(average_reward)
            delta_i = math.sqrt(3/2*math.log(n + 1)/numbers_of_selections[i])
            delta_i = float(delta_i)
            upper_bound = average_reward + delta_i
            upper_bound = float(upper_bound)
            
        else:
            upper_bound = 1e400
            upper_bound = np.float64(upper_bound)
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            max_upper_bound = np.float64(max_upper_bound)
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    reward = np.float64(reward)
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    total_reward = np.int64(total_reward)
    
#Visualizing the Results
plt.hist(ads_selected)
plt.title('Histogram of Selections')
plt.xlabel('Ads')
plt.ylabel('Count of each Ad Selection')