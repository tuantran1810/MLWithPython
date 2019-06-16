import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"

X, y = mglearn.datasets.make_wave(n_samples = 100)
line = np.linspace(-3, 3, 1000, endpoint = False).reshape(-1, 1)

print(splitStr)
bins = np.linspace(-3, 3, 11)
which_bin = np.digitize(X, bins = bins)

encoder = OneHotEncoder(sparse = False)
encoder.fit(which_bin)

X_binned = encoder.transform(which_bin)

X_combined = np.hstack([X, X_binned])
print("X_combined.shape: ", X_combined.shape)

reg = LinearRegression().fit(X_combined, y)

line_binned = encoder.transform(np.digitize(line, bins = bins))
line_combined = np.hstack([line, line_binned])

plt.plot(line, reg.predict(line_combined), label = 'linear regression combined')

for bin in bins:
	plt.plot([bin, bin], [-3, 3], ':', c = 'k', linewidth = 1)

plt.legend(loc = "best")
plt.xlabel("regression output")
plt.ylabel("input feature")
plt.plot(X[:, 0], y, 'o', c = 'k')

# plt.figure()
X_product = np.hstack([X_binned, X * X_binned])
print("X_product.shape: ", X_product.shape)

reg = LinearRegression().fit(X_product, y)
line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label = 'linear regression product')

for bin in bins:
	plt.plot([bin, bin], [-3, 3], ':', c = 'k', linewidth = 1)

plt.plot(X[:, 0], y, 'o', c = 'k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc = 'best')

plt.show()
