from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state = 1)

print("size of training set: {}".format(len(X_train)))
print("size of validation set: {}".format(len(X_val)))
print("size of test set: {}".format(len(X_test)))

bestScore = 0
bestParameters = None

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
	for C in [0.001, 0.01, 0.1, 1, 10, 100]:
		svm = SVC(gamma = gamma, C = C).fit(X_train, y_train)
		score = svm.score(X_val, y_val)
		if score > bestScore:
			bestScore = score
			bestParameters = {'C': C, 'gamma': gamma}

print("best score: {:.3f}".format(bestScore))
print("best parameters: {}".format(bestParameters))

svm = SVC(gamma = bestParameters['gamma'], C = bestParameters['C']).fit(X_train, y_train)
print("score testing with test set: {:.3f}".format(svm.score(X_test, y_test)))
