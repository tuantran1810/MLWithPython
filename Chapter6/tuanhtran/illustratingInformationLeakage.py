from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
import numpy as np

rnd = np.random.RandomState(seed = 0)
X = rnd.normal(size = (100, 10000))
y = rnd.normal(size = (100,))

select = SelectPercentile(score_func = f_regression, percentile = 5).fit(X, y)
X_selected = select.transform(X)

print("X_selected.shape: ", X_selected.shape)

print("Cross-validation accuracy (cv only on ridge): {:.3f}".format(
	np.mean(cross_val_score(Ridge(), X_selected, y, cv = 5))))

pipe = Pipeline([("select", SelectPercentile(score_func = f_regression, percentile = 5)), ("ridge", Ridge())])

print("Cross-validation accuracy (pipeline): {:.3f}".format(np.mean(cross_val_score(pipe, X, y, cv = 5))))

