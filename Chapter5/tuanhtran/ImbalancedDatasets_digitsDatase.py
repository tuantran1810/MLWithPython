from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import mglearn
import matplotlib.pyplot as plt

digits = load_digits()
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state = 0)

dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
predict_most_frequent = dummy_majority.predict(X_test)
print("unique predicted labels: {}".format(np.unique(predict_most_frequent)))
print("test set score: {:.3f}".format(dummy_majority.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)

print("Test score: {:.3f}".format(tree.score(X_test, y_test)))

dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print("dummy score: {:.3f}".format(dummy.score(X_test, y_test)))

logreg = LogisticRegression(C = 0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("logreg score: {:.3f}".format(logreg.score(X_test, y_test)))

confusion = confusion_matrix(y_test, pred_logreg)
print("logreg confusion matrix:\n{}".format(confusion))

confusion = confusion_matrix(y_test, predict_most_frequent)
print("predict_most_frequent confusion matrix:\n{}".format(confusion))

confusion = confusion_matrix(y_test, pred_tree)
print("pred_tree confusion matrix:\n{}".format(confusion))

confusion = confusion_matrix(y_test, pred_dummy)
print("pred_dummy confusion matrix:\n{}".format(confusion))

print("f1_score predict_most_frequent: {:.3f}".format(f1_score(y_test, predict_most_frequent)))
print("f1_score pred_dummy: {:.3f}".format(f1_score(y_test, pred_dummy)))
print("f1_score pred_tree: {:.3f}".format(f1_score(y_test, pred_tree)))
print("f1_score pred_logreg: {:.3f}".format(f1_score(y_test, pred_logreg)))
print(classification_report(y_test, predict_most_frequent, target_names = ["not nine", "nine"]))
print(classification_report(y_test, pred_dummy, target_names = ["not nine", "nine"]))
print(classification_report(y_test, pred_logreg, target_names = ["not nine", "nine"]))

# mglearn.plots.plot_confusion_matrix_illustration()
# plt.figure()
# mglearn.plots.plot_binary_confusion_matrix()
# plt.show()
