import mglearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

X, y = make_moons(n_samples = 200, noise = 0.05, random_state = 0)

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()

clusters = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c = clusters, cmap = mglearn.cm2, s = 60)
plt.show()
