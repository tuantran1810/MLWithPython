import mglearn
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"
X, y = make_moons(n_samples = 200, noise = 0.05, random_state = 0)

kmeans = KMeans(n_clusters = 10, random_state = 0).fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c = y_pred, s = 60, cmap = 'Paired')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 60,
	marker = '^', c = range(kmeans.n_clusters), linewidth = 2, cmap = 'Paired')

plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

print(splitStr)

distance_features = kmeans.transform(X)
print("Distance feature shape: {}".format(distance_features.shape))
print("Distance feature: \n{}".format(distance_features))
