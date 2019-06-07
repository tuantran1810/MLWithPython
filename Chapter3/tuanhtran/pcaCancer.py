from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mglearn
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"

fig, axes = plt.subplots(15, 2, figsize = (10, 20))
cancer = load_breast_cancer()
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

ax = axes.ravel()

for i in range(30):
	_, bins = np.histogram(cancer.data[:, i], bins = 50)
	ax[i].hist(malignant[:, i], bins = bins, color = mglearn.cm3(0), alpha = 0.5)
	ax[i].hist(benign[:, i], bins = bins, color = mglearn.cm3(2), alpha = 0.5)
	ax[i].set_title(cancer.feature_names[i])
	ax[i].set_yticks(())

ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignant", "benign"], loc = "best")
fig.tight_layout()
# plt.show()

scaler = StandardScaler().fit(cancer.data)
X_scaled = scaler.transform(cancer.data)
pca = PCA(n_components = 2).fit(X_scaled)
X_pca = pca.transform(X_scaled)

plt.figure(figsize = (8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc = "best")
plt.gca().set_aspect("equal")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
# plt.show()

print(splitStr)
print("PCA component shape: {}".format(pca.components_.shape))
print("PCA components: \n{}".format(pca.components_))

plt.matshow(pca.components_, cmap = 'viridis')
plt.yticks([0, 1], ["1st component", "2nd component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names,
	rotation = 60, ha = 'left')
plt.xlabel("Feature")
plt.ylabel("Principle components")
plt.show()
