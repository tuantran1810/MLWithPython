import mglearn
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

print(splitStr)
print("Lasso 1.0")
lasso = Lasso().fit(X_train, y_train)
print("Lasso training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Lasso test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of feature used: ", np.sum(lasso.coef_ != 0))
print("Lasso intercept: ", lasso.intercept_)

print(splitStr)
print("Lasso 0.01")
lasso001 = Lasso(alpha = 0.01, max_iter = 100000).fit(X_train, y_train)
print("Lasso training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Lasso test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of feature used: ", np.sum(lasso001.coef_ != 0))
print("Lasso intercept: ", lasso001.intercept_)

print(splitStr)
print("Lasso 0.0001")
lasso00001 = Lasso(alpha = 0.0001, max_iter = 100000).fit(X_train, y_train)
print("Lasso training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Lasso test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of feature used: ", np.sum(lasso00001.coef_ != 0))
print("Lasso intercept: ", lasso00001.intercept_)

plt.plot(lasso.coef_, 's', label = "Lasso alpha = 1.0")
plt.plot(lasso001.coef_, '^', label = "Lasso alpha = 0.01")
plt.plot(lasso00001.coef_, 'v', label = "Lasso alpha = 0.0001")

plt.legend()
plt.ylim(-25, 25)
plt.xlabel("coefficient index")
plt.ylabel("coefficient magnitude")
plt.show()
