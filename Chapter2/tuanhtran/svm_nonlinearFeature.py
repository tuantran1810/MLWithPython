import sklearn
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D, axes3d

X, y = make_blobs(centers = 4, random_state = 8)
y = y % 2

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
# plt.show()

linearSVM = LinearSVC().fit(X, y)
mglearn.plots.plot_2d_separator(linearSVM, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
# plt.show()

X_new = np.hstack([X, X[:, 1:] ** 2])
figure = plt.figure()
ax = Axes3D(figure, elev = -152, azim = -26)

mask = y == 0

ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c = 'b',
	cmap = mglearn.cm2, s = 60, edgecolor = 'k')

ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c = 'r',
	cmap = mglearn.cm2, s = 60, edgecolor = 'k')
ax.set_xlabel("Feature 0")
ax.set_ylabel("Feature 1")
ax.set_zlabel("Feature 1 ** 2")
plt.show()

linearSVM3D = LinearSVC().fit(X_new, y)
coef, intercept = linearSVM3D.coef_.ravel(), linearSVM3D.intercept_
figure = plt.figure()
ax = Axes3D(figure, elev = -152, azim = -26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = -(coef[0]*XX + coef[1]*YY + intercept) / coef[2]

ax.plot_surface(XX, YY, ZZ, rstride = 8, cstride = 8, alpha = 0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c = 'b',
	cmap = mglearn.cm2, s = 60, edgecolor = 'k')

ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c = 'r',
	cmap = mglearn.cm2, s = 60, edgecolor = 'k')

ax.set_xlabel("Feature 0")
ax.set_ylabel("Feature 1")
ax.set_zlabel("Feature 1 ** 2")
plt.show()

ZZ = YY ** 2
dec = linearSVM3D.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels = [dec.min(), 0, dec.max()],
	cmap = mglearn.cm2, alpha = 0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
ax.set_xlabel("Feature 0")
ax.set_ylabel("Feature 1")
plt.show()
