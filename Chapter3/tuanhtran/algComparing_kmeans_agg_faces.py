from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward
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

kmeans = KMeans(n_clusters = 10, random_state = 0)
labels_km = kmeans.fit_predict(X_pca)
print("unique label: {}".format(np.unique(labels_km)))
print("number of points per cluster: {}".format(np.bincount(labels_km)))

print("kmeans.cluster_centers_.shape: ", kmeans.cluster_centers_.shape)

fig, axes = plt.subplots(2 ,5, subplot_kw = {'xticks': (), 'yticks': ()}, figsize = (12, 4))

for center, ax in zip(kmeans.cluster_centers_, axes.ravel()):
	ax.imshow(pca.inverse_transform(center).reshape(image_shape), vmin = 0, vmax = 1)

mglearn.plots.plot_kmeans_faces(kmeans, pca, X_pca, X_people, y_people, people.target_names)

print(splitStr)
agglomerative = AgglomerativeClustering(n_clusters = 10)
labels_agg = agglomerative.fit_predict(X_pca)
print("unique label: {}".format(np.unique(labels_agg)))
print("number of points per cluster: {}".format(np.bincount(labels_agg)))
print("ARI between Kmeans and Agg: {}".format(adjusted_rand_score(labels_agg, labels_km)))

linkageArr = ward(X_pca)
plt.figure(figsize = (20, 5))
dendrogram(linkageArr, p = 7, truncate_mode = 'level', no_labels = True)
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")

n_clusters = 10
for cluster in range(n_clusters):
	mask = labels_agg == cluster
	fig, axes = plt.subplots(1, 10, subplot_kw = {'xticks': (), 'yticks': ()}, figsize = (15, 8))
	axes[0].set_ylabel(np.sum(mask))
	for image, label, asdf, ax in zip(X_people[mask], y_people[mask], labels_agg[mask], axes):
		ax.imshow(image.reshape(image_shape), vmin = 0, vmax = 1)
		ax.set_title(people.target_names[label].split()[-1], fontdict = {'fontsize': 9})

plt.show()
