import mglearn
import matplotlib.pyplot as plt

splitStr = "\n" + "=" * 150 + "\n"

X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc = 4)
plt.xlabel("First feature")
plt.ylabel("Second feature")

print(splitStr)
print("X.shape: ", X.shape)
print("y.shape: ", y.shape)
print(splitStr)
print("X:")
print(X)
print(splitStr)
print("y:")
print(y)

plt.show()
