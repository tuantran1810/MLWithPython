from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
import numpy as np

splitStr = "\n" + "=" * 150 + "\n"

irisDataset = load_iris()

print("load Iris done!")

irisKeys = irisDataset.keys()

for k in irisKeys:
	print(splitStr)
	print("Iris " + k)
	print(irisDataset[k])

X_train, X_test, y_train, y_test = train_test_split(irisDataset['data'], irisDataset['target'], random_state = 0, stratify = irisDataset['target'])

print(splitStr)
print("X_train: ")
print(X_train)

print(splitStr)
print("X_test: ")
print(X_test)

print(splitStr)
print("y_train: ")
print(y_train)

print(splitStr)
print("y_test: ")
print(y_test)

print(splitStr)
print("train size: ", X_train.shape)
print("test size: ", X_test.shape)

irisDataFrame = pd.DataFrame(X_train, columns = irisDataset.feature_names)
pd.plotting.scatter_matrix(irisDataFrame, c = y_train, figsize = (10, 10), marker = 'o', hist_kwds = {'bins' : 20}, s = 60, alpha = 0.5, cmap = mglearn.cm3, grid = True)

plt.show()

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
print(knn)

y_predict = knn.predict(X_test)
print("predict result: ")
print(y_predict)
print("actual result: ")
print(y_test)

print("test set score: " + str(np.mean(y_predict == y_test)))
# print(np.mean(y_train))
# print(np.mean(y_test))
# print(np.var(y_train))
# print(np.var(y_test))

print(np.mean(X_train[:,0]))
print(np.mean(X_test[:,0]))
print(np.var(X_train[:,0]))
print(np.var(X_test[:,0]))
