from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

splitStr = "\n" + "=" * 100 + "\n"

iris = load_iris()
logreg = LogisticRegression()

kfold = KFold(n_splits = 5)
print("cross validation scores n_splits = 5:\n{}".format(
	cross_val_score(logreg, iris.data, iris.target, cv = kfold)))

kfold = KFold(n_splits = 3)
print("cross validation scores n_splits = 3:\n{}".format(
	cross_val_score(logreg, iris.data, iris.target, cv = kfold)))

kfold = KFold(n_splits = 3, shuffle = True, random_state = 0)
print("cross validation scores n_splits = 3 (stratified):\n{}".format(
	cross_val_score(logreg, iris.data, iris.target, cv = kfold)))
