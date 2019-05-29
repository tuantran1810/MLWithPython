import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

splitStr = "\n" + "=" * 100 + "\n"

X, y = mglearn.datasets.make_wave(n_samples = 40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
reg = KNeighborsRegressor(n_neighbors = 3)
reg.fit(X_train, y_train)

print(splitStr)
print("Test set prediction: " + str(reg.predict(X_test)) + "\n")
print("Test target: " + str(y_test) + "\n")
print("Test set R^2: " + str(reg.score(X_test, y_test)))

print(splitStr)
print("Train set prediction: " + str(reg.predict(X_train)) + "\n")
print("Train target: " + str(y_train))
print(splitStr)
