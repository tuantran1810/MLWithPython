from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
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

select = SelectPercentile(percentile = 50).fit(X_train, y_train)

X_train_selected = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))

mask = select.get_support()
print("Mask: \n{}".format(mask))

plt.matshow(mask.reshape(1, -1), cmap = 'gray_r')
plt.xlabel("sample index")
plt.yticks(())
plt.show()

X_test_selected = select.transform(X_test)

lr = LogisticRegression().fit(X_train, y_train)
print("score without feature selection: {:.3f}".format(lr.score(X_test, y_test)))

lr.fit(X_train_selected, y_train)
print("score with feature selection: {:.3f}".format(lr.score(X_test_selected, y_test)))
