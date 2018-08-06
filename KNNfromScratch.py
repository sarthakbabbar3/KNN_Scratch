# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 01:09:37 2018

@author: SARTHAK BABBAR
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
from math import sqrt

style.use('fivethirtyeight')

#euclidean_distance=sqrt((plot1[0]-plot2[0])**2 +(plot1[1]-plot2[1])**2)
#class is K
data={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

# i for k or r , ii for pairs
#or [plt.scatter(ii[0],ii[1], s=100, color =i)  for ii in dataset[i] for i in dataset]

#for i in dataset:
 #   for ii in dataset[i]:
  #      plt.scatter(ii[0],ii[1], s=100, color =i)
        
#plt.scatter(new_features[0],new_features[1])
#plt.show()
#length of dictionary is k
#np.sqrt((np.array(features)  - np.array(predict))**2)
def k_nearest_neighbors(data,predict,k=3):
    if len(data) >=k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance= np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
             
    votes=[i[1] for i in sorted(distances)[:k]]
    #contains sorted classes since we are calculating i[1] which is class and i[0] is distance
    print(Counter(votes).most_common(1))
    votes_result = Counter(votes).most_common(1)[0][0]
    
    
    
    return votes_result

result = k_nearest_neighbors(data,new_features,k=3)
print(result)
    

for i in data:
    for ii in data[i]:
        plt.scatter(ii[0],ii[1], s=100, color =i)
        
plt.scatter(new_features[0],new_features[1], color = result)
plt.show()
    
