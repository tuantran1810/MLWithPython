from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import mglearn

X, _ = make_blobs(n_samples = 50, centers = 5, random_state = 4, cluster_std = 2)
X_train, X_test = train_test_split(X, random_state = 5, test_size = 0.1)

fig, axes = plt.subplots(1, 3, figsize = (13, 4))

axes[0].scatter(X_train[:, 0], X_train[:, 1], c = mglearn.cm2(0),
	label = "Training set", s = 60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], c = mglearn.cm2(1),
	label = "Test set", s = 60, marker = '^')
axes[0].legend(loc = 'upper left')
axes[0].set_title("Original data")

scaler = MinMaxScaler().fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

axes[1].scatter(X_train_scale[:, 0], X_train_scale[:, 1], c = mglearn.cm2(0),
	label = "Training set", s = 60)
axes[1].scatter(X_test_scale[:, 0], X_test_scale[:, 1], c = mglearn.cm2(1),
	label = "Test set", s = 60, marker = '^')
axes[1].set_title("Scaled data")

test_scaler = MinMaxScaler().fit(X_test)
X_test_scale_badly = test_scaler.transform(X_test)

axes[2].scatter(X_train_scale[:, 0], X_train_scale[:, 1], c = mglearn.cm2(0),
	label = "Training set", s = 60)
axes[2].scatter(X_test_scale_badly[:, 0], X_test_scale_badly[:, 1], c = mglearn.cm2(1),
	label = "Test set", s = 60, marker = '^')
axes[2].set_title("Improperly Scaled data")

plt.show()
