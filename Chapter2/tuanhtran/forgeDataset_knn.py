import sklearn 
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

splitStr = "\n" + "=" * 100 + "\n"

X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# print(splitStr)
# print("X: ")
# print(X)
# print(splitStr)
# print("y: ")
# print(y)
# print(splitStr)
# print("X_train: ")
# print(X_train)
# print(splitStr)
# print("X_test: ")
# print(X_test)
# print(splitStr)
# print("y_train: ")
# print(y_train)
# print(splitStr)
# print("y_test: ")
# print(y_test)

print(splitStr)
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
print("classifier model: " + str(clf))
print("testing with X_test: " + str(clf.predict(X_test)))
print("comparing to y_test: " + str(y_test))
print("accuracy score: " + str(clf.score(X_test, y_test)))

print(splitStr)
fig, axes = plt.subplots(1, 3, figsize = (10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
	clf = KNeighborsClassifier(n_neighbors = n_neighbors).fit(X, y)
	mglearn.plots.plot_2d_separator(clf, X, fill = True, eps = 0.5, ax = ax, alpha = 0.4)
	mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax = ax)
	ax.set_title("{} neighbor(s)".format(n_neighbors))
	ax.set_xlabel("feature 0")
	ax.set_ylabel("feature 1")
axes[0].legend(loc = 3)
plt.show()

