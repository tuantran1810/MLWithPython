from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import pandas as pd

splitStr = "\n" + "=" * 100 + "\n"

iris = load_iris()

param_grid = { 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100] }

scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv = 5), iris.data, iris.target, cv = 5)
print(splitStr)
print("Cross validation score: {}".format(scores))
print("Average cross validation score: {:.3f}".format(scores.mean()))
