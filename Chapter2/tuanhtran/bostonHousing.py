import sklearn
from sklearn.datasets import load_boston

splitStr = "\n" + "=" * 100 + "\n"

print(splitStr)
boston = load_boston()
keys = boston.keys()
print("keys: " + str(keys))
print("data shape: " + str(boston.data.shape))
print(splitStr)

for k in keys:
	print(splitStr)
	print("boston " + k)
	print(boston[k])

