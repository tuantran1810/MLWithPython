import sklearn
from sklearn.datasets import load_breast_cancer

splitStr = "\n" + "=" * 100 + "\n"

cancer = load_breast_cancer()

print(splitStr)
print("cancer keys: ")
keys = cancer.keys()
print(keys)
print("cancer shape: " + str(cancer.data.shape))

print(splitStr)
for k in keys:
	print(splitStr)
	print("cancer " + k)
	print(cancer[k])
