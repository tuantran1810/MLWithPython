import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

splitStr = "\n" + "=" * 100 + "\n"

X, y = mglearn.datasets.make_wave(n_samples = 40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
reg = KNeighborsRegressor(n_neighbors = 3)
reg.fit(X_train, y_train)

print(splitStr)
print("Test set prediction: " + str(reg.predict(X_test)) + "\n")
print("Test target: " + str(y_test) + "\n")
print("Test set R^2: " + str(reg.score(X_test, y_test)))

print(splitStr)
print("Train set prediction: " + str(reg.predict(X_train)) + "\n")
print("Train target: " + str(y_train))
print(splitStr)

fig, axes = plt.subplots(1, 3, figsize = (15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
	reg = KNeighborsRegressor(n_neighbors = n_neighbors)
	reg.fit(X_train, y_train)
	ax.plot(line, reg.predict(line))
	ax.plot(X_train, y_train, '^', c = mglearn.cm2(0), markersize = 8)
	ax.plot(X_test, y_test, 'v', c = mglearn.cm2(1), markersize = 8)
	ax.set_title("{} neighbor(s)\n train score: {:.2f}, test score: {:.2f}".
		format(n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
	ax.set_xlabel("Feature")
	ax.set_ylabel("Target")
axes[0].legend(["Model prediction", "Training data/target", "Test data/target"], loc = "best")

plt.show()
