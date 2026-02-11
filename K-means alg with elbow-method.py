from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# User can import a csv file to create the dataset for clustering.
df = pd.read_csv('your_dataset.csv')

# We get the values from the dataframe to use in clustering.
X = df.values

# Function to determine the optimal number of clusters using the elbow method.
def elbow_method(X, max_h):
    distortions = []
    K = range(1, max_h + 1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42)
        kmeanModel.fit(X)
        distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    plt.show()

# Now that we have determined the optimal number of clusters, we can define a variable to hold that value.
# I have entered a random value for max_h, you can change it based on your dataset.
n_clusters = elbow_method(X, 10)

# Now we can implement K-means clustering.
# Function to implement K-means clustering and return the labels and cluster centers.
def kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_

# Display the results of clustering.
labels, centers = kmeans_clustering(X, n_clusters)
print("Cluster Centers:\n", centers)
print("Cluster Labels:\n", labels)
