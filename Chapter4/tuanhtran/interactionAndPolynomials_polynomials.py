import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"

X, y = mglearn.datasets.make_wave(n_samples = 100)
poly = PolynomialFeatures(degree = 10, include_bias = False).fit(X)
line = np.linspace(-3, 3, 1000, endpoint = False).reshape(-1, 1)

X_poly = poly.transform(X)

print(splitStr)
print("X_poly.shape: ", X_poly.shape)
print("Entries of X: \n{}".format(X[:5]))
print("Entries of X_poly: \n{}".format(X_poly[:5]))
print("Polynomial feature names: \n{}".format(poly.get_feature_names()))

reg = LinearRegression().fit(X_poly, y)
line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label = 'polynomials linear regression')
plt.plot(X[:, 0], y, 'o', c = 'k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc = "best")

plt.figure()

for gamma in [1, 10]:
	svr = SVR(gamma = gamma).fit(X, y)
	plt.plot(line, svr.predict(line), label = 'SVR gamma = {}'.format(gamma))

plt.plot(X[:, 0], y, 'o', c = 'k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc = "best")

plt.show()
