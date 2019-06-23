from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import mglearn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve, f1_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X, y = make_blobs(n_samples = (4000, 500), cluster_std = [7.0, 2], random_state = 22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

svc = SVC(gamma = 0.05).fit(X_train, y_train)

precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))

close_zero = np.argmin(np.abs(thresholds))
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize = 10,
	label = "threshold zero", fillstyle = "none", c = 'k', mew = 2)

print(precision.shape)
print(recall.shape)
print(thresholds.shape)

# print(precision[:5])
# print(recall[:5])
# print(thresholds[:5])

plt.plot(precision, recall, label = "precision - recall curve")

rf = RandomForestClassifier(n_estimators = 100, random_state = 0, max_features = 2).fit(X_train, y_train)
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])

close_zero = np.argmin(np.abs(thresholds_rf - 0.5))

plt.plot(precision_rf, recall_rf, label = "rf curve")
plt.plot(precision_rf[close_zero], recall_rf[close_zero], '^', markersize = 10,
	label = "threshold 0.5 rf", fillstyle = "none", c = 'k', mew = 2)

plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc = "best")

print("f1_score of random forest: {:.3f}".format(f1_score(y_test, rf.predict(X_test))))
print("f1_score of svc: {:.3f}".format(f1_score(y_test, svc.predict(X_test))))

ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))

print("average_precision_score of random forest: {:.3f}".format(ap_rf))
print("average_precision_score of svc: {:.3f}".format(ap_svc))

plt.show()
