from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, 
	iris.target, random_state = 42)

gbrt = GradientBoostingClassifier(learning_rate = 0.01, 
	random_state = 0).fit(X_train, y_train)

print(splitStr)
decisionFunc = gbrt.decision_function(X_test)
print("Decision function shape: ", decisionFunc.shape)
print("Max argument of decision function: \n", np.argmax(decisionFunc, axis = 1))
print("Prediction: \n", gbrt.predict(X_test))

print(splitStr)
predictProba = gbrt.predict_proba(X_test)
print("Predict proba shape: ", predictProba.shape)
print("Predict proba sum across axis 1: \n", np.sum(predictProba, axis = 1))
print("Argmax of predictProba: \n", np.argmax(predictProba, axis = 1))
print("Prediction: \n", gbrt.predict(X_test))

named_target = iris.target_names[y_train]
logreg = LogisticRegression().fit(X_train, named_target)
print(splitStr)
print("Unique classes in training data: \n", logreg.classes_)
print("Prediction: \n", logreg.predict(X_test))
decisionFunc = logreg.decision_function(X_test)
argmaxDecFunc = np.argmax(decisionFunc, axis = 1)
print("argmaxDecFunc: \n", argmaxDecFunc)
print("argmax combined with classes: \n", logreg.classes_[argmaxDecFunc])
