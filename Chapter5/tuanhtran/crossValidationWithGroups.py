from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import mglearn

X, y = make_blobs(n_samples = 12, random_state = 0)
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
logreg = LogisticRegression()

scores = cross_val_score(logreg, X, y, groups, cv = GroupKFold(n_splits = 3))

print("Cross validation scores: {}".format(scores))
mglearn.plots.plot_group_kfold()

plt.show()
