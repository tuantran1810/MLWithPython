import mglearn
from sklearn.datasets import make_blobs, make_moons, fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"

people = fetch_lfw_people(min_faces_per_person = 20, resize = 0.7)
image_shape = people.images[0].shape

print(splitStr)
mask = np.zeros(people.target.shape, dtype = np.bool)
for target in np.unique(people.target):
	mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask] / 255
y_people = people.target[mask]

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people,
	stratify = y_people, random_state = 0)

nmf = NMF(n_components = 100, random_state = 0).fit(X_train)
pca = PCA(n_components = 100, random_state = 0).fit(X_train)
kmeans = KMeans(n_clusters = 100, random_state = 0).fit(X_train)

X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_nmf = pca.inverse_transform(nmf.transform(X_test))
X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]

fig, axes = plt.subplots(3, 5, figsize = (8, 8), subplot_kw = {'xticks': (), 'yticks': ()})
fig.suptitle("Extracted components")

for ax, comp_kmeans, comp_pca, comp_nmf in zip(axes.T, kmeans.cluster_centers_,
	pca.components_, nmf.components_):
	ax[0].imshow(comp_kmeans.reshape(image_shape))
	ax[1].imshow(comp_pca.reshape(image_shape), cmap = 'viridis')
	ax[2].imshow(comp_nmf.reshape(image_shape))

axes[0, 0].set_ylabel("kmeans")
axes[1, 0].set_ylabel("pca")
axes[2, 0].set_ylabel("nmf")

fig, axes = plt.subplots(4, 5, figsize = (8, 8), subplot_kw = {'xticks': (), 'yticks': ()})
fig.suptitle("Reconstruction")

for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(axes.T, X_test, X_reconstructed_kmeans,
	X_reconstructed_pca, X_reconstructed_nmf):
	ax[0].imshow(orig.reshape(image_shape))
	ax[1].imshow(rec_kmeans.reshape(image_shape))
	ax[2].imshow(rec_pca.reshape(image_shape))
	ax[3].imshow(rec_nmf.reshape(image_shape))

axes[0, 0].set_ylabel("original")
axes[1, 0].set_ylabel("kmeans")
axes[2, 0].set_ylabel("pca")
axes[3, 0].set_ylabel("nmf")

plt.show()
