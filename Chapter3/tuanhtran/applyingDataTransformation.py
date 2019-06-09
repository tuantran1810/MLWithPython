from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
	cancer.data, cancer.target, random_state = 1)

print(X_train.shape)
print(X_test.shape)

scaler = MinMaxScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_test_scaled perfeature min: \n{}".format(
	X_test_scaled.min(axis = 0)))
print("X_test_scaled perfeature max: \n{}".format(
	X_test_scaled.max(axis = 0)))
