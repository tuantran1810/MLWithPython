from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import mglearn

splitStr = "\n" + "=" * 100 + "\n"

def plotModelFeatureImportances(data, model):
	n_features = len(model.feature_importances_)
	plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
	plt.yticks(np.arange(n_features), data.feature_names)
	plt.xlabel("Feature importance")
	plt.ylabel("Feature")
	plt.ylim(-1, n_features)
	plt.show()

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0)

gbrt0 = GradientBoostingClassifier(random_state = 0).fit(X_train, y_train)
gbrt1 = GradientBoostingClassifier(random_state = 0, max_depth = 1).fit(X_train, y_train)
gbrt2 = GradientBoostingClassifier(random_state = 0, learning_rate = 0.01).fit(X_train, y_train)

print(splitStr)
print("Training set accuracy for gbrt0: {}".format(gbrt0.score(X_train, y_train)))
print("Test set accuracy for gbrt0: {}".format(gbrt0.score(X_test, y_test)))

print(splitStr)
print("Training set accuracy for gbrt1: {}".format(gbrt1.score(X_train, y_train)))
print("Test set accuracy for gbrt1: {}".format(gbrt1.score(X_test, y_test)))

print(splitStr)
print("Training set accuracy for gbrt2: {}".format(gbrt2.score(X_train, y_train)))
print("Test set accuracy for gbrt2: {}".format(gbrt2.score(X_test, y_test)))

plotModelFeatureImportances(cancer, gbrt0)
plotModelFeatureImportances(cancer, gbrt1)
plotModelFeatureImportances(cancer, gbrt2)
