from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import mglearn
import matplotlib.pyplot as plt

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state = 0)
lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
print("Accuracy: {:.3f}".format(accuracy_score(y_test, pred)))
print("Confusion matrix:\n{}".format(confusion_matrix(y_test, pred)))

scoreImage = mglearn.tools.heatmap(confusion_matrix(y_test, pred), xlabel = 'Predicted label', ylabel = 'True label',
	xticklabels = digits.target_names, yticklabels = digits.target_names, cmap = plt.cm.gray_r, fmt = "%d")
plt.title("Confusion matrix")
plt.gca().invert_yaxis()

print(classification_report(y_test, pred))
print("Micro avg f1 score: {:.3f}".format(f1_score(y_test, pred, average = "micro")))
print("Macro avg f1 score: {:.3f}".format(f1_score(y_test, pred, average = "macro")))

plt.show()
