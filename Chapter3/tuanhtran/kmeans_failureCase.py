import mglearn
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"

X_varied, y_varied = make_blobs(n_samples = 200, cluster_std = [1.0, 2.5, 0.5],
	random_state = 170)

kmeans = KMeans(n_clusters = 3, random_state = 0).fit(X_varied)
y_pred = kmeans.predict(X_varied)

mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
	[0, 1, 2], markers = '^', markeredgewidth = 4)
plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc = 'best')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.figure()

X, y = make_blobs(random_state = 170, n_samples = 600)
rng = np.random.RandomState(74)
transformation = rng.normal(size = (2, 2))
X = np.dot(X, transformation)

kmeans = KMeans(n_clusters = 3).fit(X)
y_pred = kmeans.predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y_pred, markers = 'o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
	[0, 1, 2], markers = '^', markeredgewidth = 4)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.figure()

X, Y = make_moons(n_samples = 200, noise = 0.05, random_state = 0)
kmeans = KMeans(n_clusters = 2).fit(X)
y_pred = kmeans.predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y_pred, markers = 'o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
	[0, 1], markers = '^', markeredgewidth = 4)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.show()
