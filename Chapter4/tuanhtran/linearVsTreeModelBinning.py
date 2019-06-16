import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"

X, y = mglearn.datasets.make_wave(n_samples = 100)
line = np.linspace(-3, 3, 1000, endpoint = False).reshape(-1, 1)

reg = DecisionTreeRegressor(min_samples_split = 3).fit(X, y)
plt.plot(line, reg.predict(line), label = "decision tree")

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label = "linear regression")

plt.plot(X[:, 0], y, 'o', c = 'k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc = "best")
plt.figure()

print(splitStr)
bins = np.linspace(-3, 3, 11)
print("bins: {}".format(bins))

which_bin = np.digitize(X, bins = bins)
print("\nData points:\n", X[:5])
print("\nBin membership for data points:\n", which_bin[:5])

encoder = OneHotEncoder(sparse = False)
encoder.fit(which_bin)

X_binned = encoder.transform(which_bin)
print(X_binned[:5])
print("X_binned.shape: {}".format(X_binned.shape))

line_binned = encoder.transform(np.digitize(line, bins = bins))

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label = 'linear regression binned')

reg = DecisionTreeRegressor(min_samples_split = 3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label = 'decision tree binned')
plt.plot(X[:, 0], y, 'o', c = 'k')
plt.vlines(bins, -3, 3, linewidth = 1, alpha = 0.2)
plt.legend(loc = "best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")

plt.show()
