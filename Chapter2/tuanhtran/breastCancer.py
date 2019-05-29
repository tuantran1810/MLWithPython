import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

splitStr = "\n" + "=" * 100 + "\n"

cancer = load_breast_cancer()

print(splitStr)
# print("cancer keys: ")
# keys = cancer.keys()
# print(keys)
# print("cancer shape: " + str(cancer.data.shape))

# print(splitStr)
# for k in keys:
# 	print(splitStr)
# 	print("cancer " + k)
# 	print(cancer[k])

X_train, X_test, y_train, y_test = train_test_split(cancer.data, 
	cancer.target, stratify = cancer.target, random_state = 66)

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
	clf = KNeighborsClassifier(n_neighbors = n_neighbors)
	clf.fit(X_train, y_train)
	training_accuracy.append(clf.score(X_train, y_train))
	test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")
plt.legend()
plt.show()
