from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
import mglearn

splitStr = "\n" + "=" * 100 + "\n"

people = fetch_lfw_people(min_faces_per_person = 20, resize = 0.7)
image_shape = people.images[0].shape

print(splitStr)
mask = np.zeros(people.target.shape, dtype = np.bool)
for target in np.unique(people.target):
	mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask] / 255
y_people = people.target[mask]

pca = PCA(n_components = 100, whiten = True, random_state = 0)
X_pca = pca.fit_transform(X_people)

dbscan = DBSCAN(min_samples = 3, eps = 15)
labels = dbscan.fit_predict(X_pca)
print("unique label: {}".format(np.unique(labels)))
print("number of points per cluster: {}".format(np.bincount(labels + 1)))

noise = X_people[labels == -1]

fig, axes = plt.subplots(4, 8, subplot_kw = {'xticks': (), 'yticks': ()}, figsize = (12, 4))
for image, ax, in zip(noise, axes.ravel()):
	ax.imshow(image.reshape(image_shape), vmin = 0, vmax = 1)

for eps in [1, 3, 5, 7, 9, 11, 13]:
	print(splitStr)
	print("eps = {}".format(eps))
	dbscan = DBSCAN(min_samples = 3, eps = eps)
	labels = dbscan.fit_predict(X_pca)
	print("unique label: {}".format(np.unique(labels)))
	print("number of points per cluster: {}".format(np.bincount(labels + 1)))

dbscan = DBSCAN(min_samples = 3, eps = 7)
labels = dbscan.fit_predict(X_pca)

for cluster in range(max(labels) + 1):
	mask = labels == cluster
	n_images = np.sum(mask)
	fig, axes = plt.subplots(1, n_images, figsize = (n_images * 1.5, 4),
		subplot_kw = {'xticks': (), 'yticks': ()})
	for image, label, ax in zip(X_people[mask], y_people[mask], axes):
		ax.imshow(image.reshape(image_shape), vmin = 0, vmax = 1)
		ax.set_title(people.target_names[label].split()[-1])

plt.show()
