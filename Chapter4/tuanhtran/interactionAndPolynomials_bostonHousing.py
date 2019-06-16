from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import mglearn

splitStr = "\n" + "=" * 100 + "\n"

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state = 0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree = 2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print(splitStr)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_poly.shape: {}".format(X_train_poly.shape))
print("Polinomial feature names:\n{}".format(poly.get_feature_names()))

print(splitStr)
print("Using Ridge Regression")
ridge = Ridge().fit(X_train_scaled, y_train)
print("score without interactions: {:.3f}".format(ridge.score(X_test_scaled, y_test)))

ridge = Ridge().fit(X_train_poly, y_train)
print("score with interactions: {:.3f}".format(ridge.score(X_test_poly, y_test)))

print(splitStr)
print("Using Random Forest Regression")
rf = RandomForestRegressor(n_estimators = 100).fit(X_train_scaled, y_train)
print("score without interactions: {:.3f}".format(rf.score(X_test_scaled, y_test)))

rf = RandomForestRegressor(n_estimators = 100).fit(X_train_poly, y_train)
print("score with interactions: {:.3f}".format(rf.score(X_test_poly, y_test)))
