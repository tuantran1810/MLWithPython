from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

splitStr = "\n" + "=" * 100 + "\n"

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
	stratify = cancer.target, random_state = 42)

for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
	lrl1 = LogisticRegression(C = C).fit(X_train, y_train)
	print(splitStr)
	print("training accuracy for C = {}: {:.2f}".format(C, lrl1.score(X_train, y_train)))
	print("test accuracy for C = {}: {:.2f}".format(C, lrl1.score(X_test, y_test)))
	plt.plot(lrl1.coef_.T, marker, label = "C = {}".format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation = 90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient Magnitude")
plt.legend(loc = 3)
plt.show()
