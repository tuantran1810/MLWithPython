from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"

def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
	outer_score = []
	for training_sample, test_sample in outer_cv.split(X, y):
		best_parms = {}
		best_score = -np.inf

		for parameters in parameter_grid:
			cv_scores = []
			for inner_train, inner_test in inner_cv.split(X[training_sample], y[training_sample]):
				clf = Classifier(**parameters).fit(X[inner_train], y[inner_train])
				score = clf.score(X[inner_test], y[inner_test])
				cv_scores.append(score)
			mean_score = np.mean(cv_scores)
			if mean_score > best_score:
				best_score = mean_score
				best_parms = parameters

		clf = Classifier(**best_parms).fit(X[training_sample], y[training_sample])
		outer_score.append(clf.score(X[test_sample], y[test_sample]))
	return np.array(outer_score)

iris = load_iris()

param_grid = { 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100] }

scores = nested_cv(iris.data, iris.target, StratifiedKFold(5), StratifiedKFold(5), SVC, ParameterGrid(param_grid))
print(splitStr)
print("Cross validation score: {}".format(scores))
print("Average cross validation score: {:.3f}".format(scores.mean()))
