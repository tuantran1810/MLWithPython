import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

splitStr = "\n" + "=" * 100 + "\n"

rnd = np.random.RandomState(0)
X_org = rnd.normal(size = (1000, 3))
w = rnd.normal(size = 3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)

print(splitStr)
print("X_org[:5]: \n{}".format(X_org[:5]))
print("X[:5]: \n{}".format(X[:5]))
print("X.shape: ", X.shape)
print("Number of feature appearances:\n{}".format(np.bincount(X[:, 0])))

bins = np.bincount(X[:, 0])
plt.bar(range(len(bins)), bins, color = 'b')
plt.ylabel("Number of feature appearances")
plt.xlabel("Value")

# plt.figure()
# bins = np.bincount(X[:, 1])
# plt.bar(range(len(bins)), bins, color = 'b')
# plt.ylabel("Number of feature appearances")
# plt.xlabel("Value")

# plt.figure()
# bins = np.bincount(X[:, 2])
# plt.bar(range(len(bins)), bins, color = 'b')
# plt.ylabel("Number of feature appearances")
# plt.xlabel("Value")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("test set score: {:.3f}".format(score))

X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

plt.figure()
plt.hist(X_train_log[:, 0], bins = 25, color = 'gray')
plt.ylabel("Number of feature appearances")
plt.xlabel("value")

plt.show()
