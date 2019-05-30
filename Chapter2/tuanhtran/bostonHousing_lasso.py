import mglearn
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

lasso = Lasso().fit(X_train, y_train)
print("Lasso training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Lasso test set score: {:.2f}".format(lasso.score(X_test, y_test)))

print("Number of feature used: ", np.sum(lasso.coef_ != 0))
print("Lasso intercept: ", lasso.intercept_)

