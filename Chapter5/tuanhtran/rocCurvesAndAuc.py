from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import mglearn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve, f1_score, average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import roc_curve

X, y = make_blobs(n_samples = (4000, 500), cluster_std = [7.0, 2], random_state = 22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
svc = SVC(gamma = 0.05).fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))

plt.plot(fpr, tpr, label = "ROC curve")

close_zero = np.argmin(np.abs(thresholds))

plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize = 10,
	label = "threshold zero", fillstyle = "none", c = 'k', mew = 2)

rf = RandomForestClassifier(n_estimators = 100, random_state = 0, max_features = 2).fit(X_train, y_train)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
plt.plot(fpr_rf, tpr_rf, label = "ROC curve")

close_zero_rf = np.argmin(np.abs(thresholds_rf - 0.5))

plt.plot(fpr_rf[close_zero_rf], tpr_rf[close_zero_rf], '^', markersize = 10,
	label = "threshold zero rf", fillstyle = "none", c = 'k', mew = 2)

rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))
print("AUC random forest: {:.3f}".format(rf_auc))
print("AUC SVC: {:.3f}".format(svc_auc))

plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
plt.legend(loc = 4)

digits = load_digits()
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state = 0)

plt.figure()
for gamma in [1, 0.05, 0.01]:
	svc = SVC(gamma = gamma).fit(X_train, y_train)
	accuracy = svc.score(X_test, y_test)
	auc = roc_auc_score(y_test, svc.decision_function(X_test))
	fpr, tpr, _ = roc_curve(y_test, svc.decision_function(X_test))
	print("gamma = {:.3f}, accuracy = {:.3f}, AUC = {:.3f}".format(gamma, accuracy, auc))
	plt.plot(fpr, tpr, label = "gamma = {:.3f}".format(gamma))

plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
plt.xlim(-0.01, 1)
plt.ylim(0, 1.02)
plt.legend(loc = "best")

plt.show()
