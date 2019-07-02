from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
import numpy as np

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 4)

pipe = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid, cv = 5).fit(X_train, y_train)
print("Best estimator:\n{}".format(grid.best_estimator_))
print("LogisticRegression step:\n{}".format(grid.best_estimator_.named_steps["logisticregression"]))
print("LogisticRegression coef:\n{}".format(grid.best_estimator_.named_steps["logisticregression"].coef_))
