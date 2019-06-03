from sklearn.svm import SVC
import mglearn
import matplotlib.pyplot as plt

splitStr = "\n" + "=" * 100 + "\n"

X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel = 'rbf', C = 10, gamma = 0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps = 0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

sv = svm.support_vectors_

sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s =15, markeredgewidth = 3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

fig, axes = plt.subplots(3, 3, figsize = (15, 10))

for ax, C in zip(axes, [-1, 0, 3]):
	for a, gamma in zip(ax, range(-1, 2)):
		mglearn.plots.plot_svm(log_C = C, log_gamma = gamma, ax = a)

axes[0, 0].legend(["class 0", "class 1", "SV class 0", "SV class 1"],
	ncol = 4, loc = (0.9, 1.2))

plt.show()
