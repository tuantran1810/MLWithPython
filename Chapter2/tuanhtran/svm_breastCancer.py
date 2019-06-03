from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
import mglearn
import matplotlib.pyplot as plt

splitStr = "\n" + "=" * 100 + "\n"

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0)

svc = SVC().fit(X_train, y_train)
print(splitStr)
print("Training set accuracy: {}".format(svc.score(X_train, y_train)))
print("Test set accuracy: {}".format(svc.score(X_test, y_test)))

plt.boxplot(X_train)
plt.yscale("symlog")
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.show()

min_on_training = X_train.min(axis = 0)
range_on_training = (X_train - min_on_training).max(axis = 0)
X_train_scale = (X_train - min_on_training) / range_on_training

X_test_scale = (X_test - min_on_training) / range_on_training

svc = SVC(C = 1000).fit(X_train_scale, y_train)
print(splitStr)
print("Training set accuracy: {}".format(svc.score(X_train_scale, y_train)))
print("Test set accuracy: {}".format(svc.score(X_test_scale, y_test)))

