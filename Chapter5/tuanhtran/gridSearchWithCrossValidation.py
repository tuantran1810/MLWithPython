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

bestScore = 0
bestParameters = None

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
	for C in [0.001, 0.01, 0.1, 1, 10, 100]:
		svm = SVC(gamma = gamma, C = C)
		score = cross_val_score(svm, X_trainval, y_trainval, cv = 5).mean()
		if score > bestScore:
			bestScore = score
			bestParameters = {'C': C, 'gamma': gamma}

print("best score: {:.3f}".format(bestScore))
print("best parameters: {}".format(bestParameters))

svm = SVC(**bestParameters).fit(X_train, y_train)
print("score testing with test set: {:.3f}".format(svm.score(X_test, y_test)))

print(splitStr)

paramGrid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

gridSearch = GridSearchCV(SVC(), paramGrid, cv = 5, return_train_score = True).fit(X_trainval, y_trainval)
print("score testing with test set: {:.3f}".format(gridSearch.score(X_test, y_test)))
print("Best parameters: ", gridSearch.best_params_)
print("Best CV score: ", gridSearch.best_score_)
print("Best estimator:\n{}".format(gridSearch.best_estimator_))
print(splitStr)

results = pd.DataFrame(gridSearch.cv_results_)
print(results)
print(splitStr)

scores = np.array(results.mean_test_score).reshape(6, 6)
mglearn.tools.heatmap(scores, xlabel = 'gamma', xticklabels = paramGrid['gamma'],
	ylabel = 'C', yticklabels = paramGrid['C'], cmap = "viridis")
plt.show()
