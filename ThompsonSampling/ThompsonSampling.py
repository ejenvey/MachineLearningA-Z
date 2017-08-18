#Thompson Sampling

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#Importing the dataset
#This dataset basically simulates an experiment where you are sending 10000 users 10 different ads, 
#and it tells us which ads the person clicks on
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implement the Thompson Sampling Algorithm (without using a library)
#number of users
N = 10000
#number of ads
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0,N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            max_random = np.float64(max_random)
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    reward = np.float64(reward)
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    total_reward = np.int64(total_reward)
    
#Visualizing the Results
plt.hist(ads_selected)
plt.title('Histogram of Selections')
plt.xlabel('Ads')
plt.ylabel('Count of each Ad Selection')
plt.show()