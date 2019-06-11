import mglearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

splitStr = "\n" + "=" * 100 + "\n"

X, y = make_blobs(random_state = 1)
kmeans = KMeans(n_clusters = 3).fit(X)

print(splitStr)
print("Cluster membership:\n{}".format(kmeans.labels_))
print(kmeans.predict(X))

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers = 'o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
	[0, 1, 2], markers = '^', markeredgewidth = 2)

fig, axes = plt.subplots(1, 2, figsize = (10, 5))

kmeans = KMeans(n_clusters = 2).fit(X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, ax = axes[0])
kmeans = KMeans(n_clusters = 5).fit(X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, ax = axes[1])

plt.show()
