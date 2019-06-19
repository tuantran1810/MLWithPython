from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
import pandas as pd

splitStr = "\n" + "=" * 100 + "\n"

iris = load_iris()
logreg = LogisticRegression()
loo = LeaveOneOut()

scores = cross_val_score(logreg, iris.data, iris.target, cv = loo)
print("cross validation points:\n{}".format(scores))
print("number of scores: ", len(scores))
print("scores mean: ", scores.mean())
