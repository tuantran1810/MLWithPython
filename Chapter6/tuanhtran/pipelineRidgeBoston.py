from sklearn.svm import SVC
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
import numpy as np
import matplotlib.pyplot as plt

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state = 0)

pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())
param_grid = {'polynomialfeatures__degree': [1, 2, 3], 'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid, cv = 5).fit(X_train, y_train)

plt.matshow(grid.cv_results_['mean_test_score'].reshape(3, -1), vmin = 0, cmap = "viridis")
plt.xlabel("ridge__alpha")
plt.ylabel("polynomialfeature_degree")
plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha'])
plt.yticks(range(len(param_grid['polynomialfeatures__degree']))), param_grid['polynomialfeatures__degree']
plt.colorbar()

print("Best parameters:\n{}".format(grid.best_params_))
print("Test set score: {:.3f}".format(grid.score(X_test, y_test)))

pipe = make_pipeline(StandardScaler(), Ridge())
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid, cv = 5).fit(X_train, y_train)
print("Test set score without polynomial: {:.3f}".format(grid.score(X_test, y_test)))

plt.show()
