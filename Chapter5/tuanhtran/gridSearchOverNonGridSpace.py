from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn

splitStr = "\n" + "=" * 100 + "\n"

print(splitStr)
iris = load_iris()
X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state = 1)

print("size of training set: {}".format(len(X_train)))
print("size of validation set: {}".format(len(X_val)))
print("size of test set: {}".format(len(X_test)))

paramGrid = [  	{
					'kernel': ['rbf'],
					'C': [0.001, 0.01, 0.1, 1, 10, 100], 
					'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
				},
				{
					'kernel': ['linear'],
					'C': [0.001, 0.01, 0.1, 1, 10, 100]
				}
			]

gridSearch = GridSearchCV(SVC(), paramGrid, cv = 5, return_train_score = True).fit(X_trainval, y_trainval)
print("score testing with test set: {:.3f}".format(gridSearch.score(X_test, y_test)))
print("Best parameters: ", gridSearch.best_params_)
print("Best CV score: ", gridSearch.best_score_)
print("Best estimator:\n{}".format(gridSearch.best_estimator_))
print(splitStr)

results = pd.DataFrame(gridSearch.cv_results_)
print(results.T)
print(splitStr)

