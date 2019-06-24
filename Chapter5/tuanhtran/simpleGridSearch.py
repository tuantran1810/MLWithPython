from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 0)

print("size of training set: {}".format(len(X_train)))
print("size of test set: {}".format(len(X_test)))

bestScore = 0
bestParameters = None

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
	for C in [0.001, 0.01, 0.1, 1, 10, 100]:
		svm = SVC(gamma = gamma, C = C).fit(X_train, y_train)
		score = svm.score(X_test, y_test)
		if score > bestScore:
			bestScore = score
			bestParameters = {'C': C, 'gamma': gamma}

print("best score: {:.3f}".format(bestScore))
print("best parameters: {}".format(bestParameters))
