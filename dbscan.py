import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"E:\B.Tech IIT Dh\Sem 2\Machine Learning\Mall Customer Segmention\Mall_Customers.csv")

#standeizing the dataset
from sklearn.preprocessing import StandardScaler
df['Gender'] = df['Gender'].map({'Male' : 0, 'Female' : 1 })
scaler = StandardScaler()
featuers = ['Age', 'Gender','Annual Income (k$)', 'Spending Score (1-100)' ]
x = df[featuers]
X_scaled = scaler.fit_transform(x)

#graph plot to find the correct epsilon value for DBSCAN 
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)

distances, indices = neighbors_fit.kneighbors(X_scaled)

distances = np.sort(distances[:,4])

plt.plot(distances)
plt.ylabel("k-distance")
plt.xlabel("Points sorted by distance")
plt.show()


#DBSCAN model training
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.8, min_samples=5).fit(X_scaled)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


#final ploting considering Spending Score and Income as axis
X_plot = df[['Spending Score (1-100)', 'Annual Income (k$)']].values

unique_labels = set(labels)

core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0,1,len(unique_labels))]

for k, col in zip(unique_labels, colors):

    if k == -1:
        col = [0,0,0,1]   # noise points

    class_member_mask = labels == k

    xy = X_plot[class_member_mask & core_samples_mask]

    plt.plot(
        xy[:,0],
        xy[:,1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14
    )

    xy = X_plot[class_member_mask & ~core_samples_mask]

    plt.plot(
        xy[:,0],
        xy[:,1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6
    )

plt.xlabel("Spending Score")
plt.ylabel("Annual Income")
plt.title(f"Estimated number of clusters: {n_clusters_}")

plt.show()