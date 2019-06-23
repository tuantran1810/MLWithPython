from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, average_precision_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
import mglearn
import matplotlib.pyplot as plt
import pandas as pd

splitStr = "\n" + "=" * 100 + "\n"

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state = 0)

print("Default scoring: ", cross_val_score(SVC(), digits.data, digits.target == 9, cv = 5))
print("Explicit accuracy: ", cross_val_score(SVC(), digits.data, digits.target == 9, cv = 5, 
	scoring = "accuracy"))
print("Average precision: ", cross_val_score(SVC(), digits.data, digits.target == 9, cv = 5, 
	scoring = "average_precision"))

res = cross_validate(SVC(), digits.data, digits.target == 9,
	scoring = ["accuracy", "average_precision", "recall_macro"],
	return_train_score = True, cv = 5)

print(pd.DataFrame(res))

print(splitStr)

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target == 9, random_state = 0)

param_grid = {'gamma': [0.0001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(SVC(), param_grid = param_grid).fit(X_train, y_train)
print("Grid search with accuracy:")
print("Best parameters: ", grid.best_params_)
print("Best cross-validation score (accuracy): {:.3f}".format(grid.best_score_))
print("Test set average precision: {:.3f}".format(average_precision_score(y_test, grid.decision_function(X_test))))
print("Test set accuracy score: {:.3f}".format(accuracy_score(y_test, grid.predict(X_test))))

print(splitStr)
grid = GridSearchCV(SVC(), param_grid = param_grid, scoring = "average_precision").fit(X_train, y_train)
print("Grid search with average precision:")
print("Best parameters: ", grid.best_params_)
print("Best cross-validation score (average precision): {:.3f}".format(grid.best_score_))
print("Test set average precision: {:.3f}".format(average_precision_score(y_test, grid.decision_function(X_test))))
print("Test set accuracy score: {:.3f}".format(accuracy_score(y_test, grid.predict(X_test))))
