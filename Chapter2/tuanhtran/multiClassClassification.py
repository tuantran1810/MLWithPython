from sklearn.datasets import make_blobs
import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from numpy import linspace

splitStr = "\n" + "=" * 100 + "\n"

X, y = make_blobs(random_state = 42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])
plt.show()

print(splitStr)
linear_svc = LinearSVC().fit(X, y)
print("linear_svc.coef_.shape: ", linear_svc.coef_.shape)
print("linear_svc.coef_: ", linear_svc.coef_)
print("linear_svc.intercept_.shape: ", linear_svc.intercept_.shape)
print("linear_svc.intercept_: ", linear_svc.intercept_)

print(splitStr)
logReg = LogisticRegression().fit(X, y)
print("logReg.coef_.shape: ", logReg.coef_.shape)
print("logReg.coef_: ", logReg.coef_)
print("logReg.intercept_.shape: ", logReg.intercept_.shape)
print("logReg.intercept_: ", logReg.intercept_)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = linspace(-15, 15)
for coef, intercept, color in zip(linear_svc.coef_, linear_svc.intercept_, mglearn.cm3.colors):
	plt.plot(line, -(line*coef[0] + intercept)/coef[1], c = color)

plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("LinearSVC")
plt.xlim(-10, 8)
plt.ylim(-10, 15)
plt.legend(["Class 0", "Class 1", "Class 2", "Line class 0", "Line class 1", "Line class 2"], loc = (1.01, 0.3))
mglearn.plots.plot_2d_classification(linear_svc, X, fill = True, alpha = 0.7)
plt.show()

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = linspace(-15, 15)
for coef, intercept, color in zip(logReg.coef_, logReg.intercept_, mglearn.cm3.colors):
	plt.plot(line, -(line*coef[0] + intercept)/coef[1], c = color)

plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("LinearRegression")
plt.xlim(-10, 8)
plt.ylim(-10, 15)
plt.legend(["Class 0", "Class 1", "Class 2", "Line class 0", "Line class 1", "Line class 2"], loc = (1.01, 0.3))
mglearn.plots.plot_2d_classification(logReg, X, fill = True, alpha = 0.7)
plt.show()
