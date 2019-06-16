from scipy.cluster.hierarchy import dendrogram, ward
import mglearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(random_state = 0, n_samples = 12)
linkageArr = ward(X)

dendrogram(linkageArr)

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c = 'k')
ax.plot(bounds, [4, 4], '--', c = 'k')

plt.show()
