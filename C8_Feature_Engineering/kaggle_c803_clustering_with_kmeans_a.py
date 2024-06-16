# Unsupervised algorithms don't make use of a target;
# instead, their purpose is to learn some property of the data, to represent the structure of the features in a certain way.
# In the context of feature engineering for prediction,
# you could think of an unsupervised algorithm as a "feature discovery" technique.

# Clustering simply means the assigning of data points to groups based upon how similar the points are to each other.
# A clustering algorithm makes "birds of a feather flock together," so to speak.

# When used for feature engineering, we could attempt to discover groups of customers representing a market segment,
# for instance, or geographic areas that share similar weather patterns.
# Adding a feature of cluster labels can help machine learning models untangle complicated relationships of space or proximity.


# Cluster Labels as a Feature
# Applied to a single real-valued feature, clustering acts like a traditional "binning" or "discretization" transform.
# On multiple features, it's like "multi-dimensional binning" (sometimes called vector quantization).


# K-means clustering measures similarity using ordinary straight-line distance (Euclidean distance, in other words).
# It creates clusters by placing a number of points, called centroids, inside the feature-space.
# Each point in the dataset is assigned to the cluster of whichever centroid it's closest to.
# The "k" in "k-means" is how many centroids (that is, clusters) it creates. You define the k yourself.

# n_clusters, max_iter, and n_init.
# It's a simple two-step process. The algorithm starts by randomly initializing some predefined number (n_clusters) of centroids.
# It then iterates over these two operations:
# 1. assign points to the nearest cluster centroid
# 2. move each centroid to minimize the distance to its points
# It iterates over these two steps until the centroids aren't moving anymore, or until some maximum number of iterations has passed (max_iter).

# It often happens that the initial random position of the centroids ends in a poor clustering.
# For this reason the algorithm repeats a number of times (n_init) and returns the clustering that
# has the least total distance between each point and its centroid, the optimal clustering.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

df = pd.read_csv("../dataset/housing.csv")
X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]
print(X.head())


# Create cluster feature
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

print(X.head())


sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
);

X["MedHouseVal"] = df["MedHouseVal"]
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6);

plt.show()