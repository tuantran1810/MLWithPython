import mglearn
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
lr = LinearRegression().fit(X_train, y_train)
ridge1 = Ridge().fit(X_train, y_train)
ridge01 = Ridge(alpha = 0.1).fit(X_train, y_train)
ridge10 = Ridge(alpha = 10).fit(X_train, y_train)

print("ridge1 training set score: {:.2f}".format(ridge1.score(X_train, y_train)))
print("ridge1 test set score: {:.2f}".format(ridge1.score(X_test, y_test)))

print("ridge01 training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("ridge01 test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

print("ridge10 training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("ridge10 test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

plt.plot(ridge1.coef_, 's', label = "Ridge alpha = 1")
plt.plot(ridge10.coef_, '^', label = "Ridge alpha = 10")
plt.plot(ridge01.coef_, 'v', label = "Ridge alpha = 0.1")\

plt.plot(lr.coef_, 'o', label = "Linear Regression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()

