from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import mglearn
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize = (10, 3))

for model, ax in zip([LogisticRegression(), LinearSVC()], axes):
	clf = model.fit(X, y)
	print(clf.coef_)
	print(clf.intercept_)
	mglearn.plots.plot_2d_separator(clf, X, fill = False, eps = 0.5 , ax = ax, alpha = 0.7)
	mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax = ax)
	ax.set_title(clf.__class__.__name__)
	ax.set_xlabel("Feature 0")
	ax.set_ylabel("Feature 1")
axes[0].legend()

mglearn.plots.plot_linear_svc_regularization()
plt.show()
