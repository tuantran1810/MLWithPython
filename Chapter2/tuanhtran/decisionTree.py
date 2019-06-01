import mglearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
import numpy as np

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
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
	stratify = cancer.target, random_state = 42)

print(splitStr)
tree = DecisionTreeClassifier(random_state = 0).fit(X_train, y_train)
print("training set score: ", tree.score(X_train, y_train))
print("test set score: ", tree.score(X_test, y_test))
export_graphviz(tree, out_file = "breastCancerTree.dot", class_names = ["malinant", "begnign"],
	feature_names = cancer.feature_names, impurity = False, filled = True)
print("feature importances: ")
print(tree.feature_importances_)
plotModelFeatureImportances(cancer, tree)

print(splitStr)
tree = DecisionTreeClassifier(random_state = 0, max_depth = 4).fit(X_train, y_train)
print("training set score for max depth limit: ", tree.score(X_train, y_train))
print("test set score for max depth limit: ", tree.score(X_test, y_test))
export_graphviz(tree, out_file = "breastCancerTreeLimit.dot", class_names = ["malinant", "begnign"],
	feature_names = cancer.feature_names, impurity = False, filled = True)
print("feature importances: ")
print(tree.feature_importances_)
plotModelFeatureImportances(cancer, tree)

# graphviz.Source.from_file("breastCancerTree.dot").view()
# graphviz.Source.from_file("breastCancerTreeLimit.dot").view()

