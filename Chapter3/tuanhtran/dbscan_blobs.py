from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state = 0, n_samples = 12)
dbscan = DBSCAN()
cluster = dbscan.fit_predict(X)
print("Cluster:\n{}".format(cluster))
