# -*- coding: utf-8 -*-
"""Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l6hNlwGIe_S9IxFDVmqsdbja6O7L2pvX
"""

# with urllib.request.urlopen('https://data.gov.il/api/3/action/datastore_search?resource_id=0d4c476d-e5bb-459c-b5b3-b4796ce63d5b&limit=5&q=title:jones') as url:
#     data = url.read()
#     print(data)

# scaler = MinMaxScaler()
# scaler.fit(data.iloc[:,1:])
# scaled_data = pd.DataFrame(scaler.transform(data.iloc[:,1:]))
# scaled_data.describe()
# scaled_data.head()
# scaled_data.describe()

# feature['Briut_ahuz_13'] = feature['Briut_ahuz_13'].replace('-', None)
# feature['Briut_ahuz_13'] = feature['Briut_ahuz_13'].fillna(feature['Briut_ahuz_13'].mode())

import pandas as pd
import numpy as np
import urllib.request
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

elections = pd.read_csv('/content/dataB.csv')
election = elections.iloc[1:,1:-1]
#df2 = pd.read_csv('/content/dataB.csv')
features = pd.read_csv('/content/dataC.csv')
names = features[["yshuv","shem_yshuv"]]
feature = features[["Totalpop16","AhuzOlim1990","Bagrut_1516","Eshkol_15","Shetah_shiput_13","DiabRate14_16"]] #'Briut_ahuz_13'

# frames = [election.iloc[1:,1:], features]
# data = pd.concat(frames)
# print(names)

election.isna().sum()

feature.isna().sum()

feature['AhuzOlim1990'] = feature['AhuzOlim1990'].fillna(0.0)

feature.describe()

election.head()

feature['label'] = feature.Totalpop16.astype(str) + '_' + feature['Bagrut_1516'].astype(str) + '_' + feature['Eshkol_15'].astype(str)
feature.head()

election['label'] = election['Symbol of a settlement'].astype(str) + '_' + election['Voters'].astype(str) + '_' + election['Kosher'].astype(str)
election.head()

# import sys
# print(sys.getrecursionlimit())

import sys
sys.setrecursionlimit(2000)

print(sys.getrecursionlimit())

"""#Hierarchical clustering Algorithm 


"""

Z_list = []
data_scaling_methods= [MinMaxScaler(),StandardScaler()]
linkage_methods = ['single','complete','average','centroid']
pictures = {}

for scaler in data_scaling_methods:
  for linkage in linkage_methods:
    plt.figure(figsize = (25,10))
    scaler.fit( election.iloc[:,1:-1])
    scaled_data = pd.DataFrame(scaler.transform( election.iloc[:2000,1:-1]))
    #print(scaled_data )
    Z = sch.linkage(scaled_data , method=linkage)
    Z_list.append(Z)
    print('The data scaling method is: ',scaler ,"and the linkage method is: ", linkage)
    dend = sch.dendrogram(Z , labels = election['label'].values, leaf_rotation=90)
    plt.show()

"""# K-Means Algorithm 



"""

# scaler = MinMaxScaler()
# scaler.fit(election.iloc[:,:-1])

SSE = []
for k in range(1,10):
  model = KMeans(n_clusters = k,init='k-means++')
  model.fit(election.iloc[:,:-1])
  SSE.append(model.inertia_)

plt.plot(range(1,10),SSE) 
plt.scatter(range(1,10),SSE,c= 'r',marker = '*') 
plt.xlabel("number of clusters") 
plt.ylabel("SSE") 
plt.grid() 
plt.show() 

kmeans_model = KMeans(n_clusters = 2) 
kmeans_model.fit(election.iloc[:,:-1])  
print("The final size of centers of {} clusters list is : {}" .format(2,kmeans_model.cluster_centers_.shape[1]) )
print("The sum square error is(SSE):", kmeans_model.inertia_) 
print("The number of iterations is:",  kmeans_model.n_iter_) 
print("*"*50)
kmeans_model = KMeans(n_clusters = 3) 
kmeans_model.fit(election.iloc[:,:-1])  
print("The final size of centers of {} clusters list is : {}" .format(3,kmeans_model.cluster_centers_.shape[1]) )
print("The sum square error is(SSE):", kmeans_model.inertia_) 
print("The number of iterations is:",  kmeans_model.n_iter_)

def rand_index(c1,c2):
  n=len(c1)
  alpha , beta, gamma, delta = 0,0,0,0
  for i in range(n-1):
    for j in range(i+1,n):
      if ((c1[i]==c1[j]) & (c2[i]==c2[j])):
        alpha += 1
      elif((c1[i]!=c1[j]) & (c2[i]!=c2[j])):
        beta += 1
      elif((c1[i]==c1[j]) & (c2[i]!=c2[j])):
        gamma += 1
      elif((c1[i]!=c1[j]) & (c2[i]==c2[j])):
        delta += 1
  #print(alpha,beta,gamma,delta)
  return (alpha+beta)/(alpha+beta+gamma+delta)

def pre_randindex():
  kmeans_model = KMeans(n_clusters = 3) 
  kmeans_model.fit(election.iloc[:,:-1])
  kmeans_clusters = kmeans_model.cluster_centers_
  scaler = MinMaxScaler()
  scaler.fit( election.iloc[:,1:-1] )
  scaled_data = pd.DataFrame(scaler.transform( election.iloc[:2000,1:-1]))
  Z = sch.linkage(scaled_data,method='complete')
  gap = []
  for i in range(1,30):   #i = num__of_clusters
          kmeans_cluster = kmeans_clusters[i%3][i]
          Hierarchical_clusters = sch.fcluster(Z_list[i],i,criterion='maxclust')
          randIndex = rand_index(Hierarchical_clusters,kmeans_cluster)
          gap = np.append(gap,randIndex)
          #print(i,randIndex)
  return gap

gap = pre_randindex()
plt.plot(1-gap)

kmeans_model = KMeans(n_clusters = 3) 
  kmeans_model.fit(election.iloc[:,:-1])
  kmeans_clusters = kmeans_model.cluster_centers_
  print(kmeans_clusters[0][0])

pd.Series(kmeans_model.labels_).unique()

Z_list[0].shape

len(election)