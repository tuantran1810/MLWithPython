import sklearn 
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

splitStr = "\n" + "=" * 100 + "\n"

X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

print(splitStr)
print("X: ")
print(X)
print(splitStr)
print("y: ")
print(y)
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
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
print("classifier model: " + str(clf))
print("testing with X_test: " + str(clf.predict(X_test)))
print("comparing to y_test: " + str(y_test))
print("accuracy score: " + str(clf.score(X_test, y_test)))
print(splitStr)
