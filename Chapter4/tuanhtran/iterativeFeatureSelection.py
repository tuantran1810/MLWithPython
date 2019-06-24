from sklearn.feature_selection import RFE
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

splitStr = "\n" + "=" * 100 + "\n"

cancer = load_breast_cancer()

rng = np.random.RandomState(42)
noise = rng.normal(size = (len(cancer.data), 50))

X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, 
	random_state = 0, test_size = 0.5)

select = RFE(RandomForestClassifier(n_estimators = 100, random_state = 42),
	n_features_to_select = 40).fit(X_train, y_train)

mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap = 'gray_r')
plt.xlabel("Sample index")
plt.yticks(())
plt.show()

X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)

logreg = LogisticRegression().fit(X_train_rfe, y_train)

print("Logreg on training set: {:.3f}".format(logreg.score(X_train_rfe, y_train)))
print("Logreg on test set: {:.3f}".format(logreg.score(X_test_rfe, y_test)))
print("RFE score: {:.3f}".format(select.score(X_test, y_test)))
