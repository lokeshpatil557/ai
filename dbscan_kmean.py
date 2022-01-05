import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
import sklearn
a=pd.read_csv("C:/Users/Lokesh/Desktop/cdac ai material/machine learning/5 jan/Mall_Customers (1).csv")
print(a.head())
a.drop('CustomerID',axis=1,inplace=True)
a.drop('Genre',axis=1,inplace=True)
a.drop('Age',axis=1,inplace=True)
print(a.info())
plt.plot(a)
plt.show()
#NEARESTNEIGHBOURS
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(a)
distances, indices = nbrs.kneighbors(a)
print(distances)
print(indices)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()


from sklearn.cluster import DBSCAN
dbscan_opt=DBSCAN(eps=6,min_samples=6)
dbscan_opt.fit(a)
a['DBSCAN_opt_labels']=dbscan_opt.labels_
a['DBSCAN_opt_labels'].value_counts()
#PLOTTING THE RESULT CLUSTERS
colors=['purple','green','red','black','blue','orange']
plt.figure(figsize=(10,10))
plt.scatter(a["Annual_Income_(k$)"],a["Spending_Score"],c=a['DBSCAN_opt_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('DBSCAN Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()







