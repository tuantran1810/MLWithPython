from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
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

agg = AgglomerativeClustering(n_clusters = 40)
labels_agg = agg.fit_predict(X_pca)

print("Cluster size: {}".format(np.bincount(labels_agg)))

n_clusters = 40

for cluster in [1, 5, 8, 17, 24]:
	mask = labels_agg == cluster
	fig, axes = plt.subplots(1, 15, subplot_kw = {'xticks': (), 'yticks': ()}, figsize = (15, 8))
	cluster_size = np.sum(mask)

	axes[0].set_ylabel("#{}: {}".format(cluster, cluster_size))
	for image, label, asdf, ax in zip(X_people[mask], y_people[mask], labels_agg[mask], axes):
		ax.imshow(image.reshape(image_shape), vmin = 0, vmax = 1)
		ax.set_title(people.target_names[label].split()[-1], fontdict = {'fontsize': 9})

	for i in range(cluster_size, 15):
		axes[i].set_visible(False)
plt.show()
