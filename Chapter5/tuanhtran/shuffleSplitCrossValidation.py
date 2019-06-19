from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

splitStr = "\n" + "=" * 100 + "\n"

iris = load_iris()
logreg = LogisticRegression()
shuffle_split = ShuffleSplit(test_size = 0.5, train_size = 0.5, n_splits = 10)

scores = cross_val_score(logreg, iris.data, iris.target, cv = shuffle_split)
print("cross validation points:\n{}".format(scores))
print("scores mean: ", scores.mean())

shuffle_split = StratifiedShuffleSplit(test_size = 0.5, train_size = 0.5, n_splits = 10)

scores = cross_val_score(logreg, iris.data, iris.target, cv = shuffle_split)
print("stratified cross validation points:\n{}".format(scores))
print("scores mean: ", scores.mean())
