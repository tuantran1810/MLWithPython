from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn
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

nmf = NMF(n_components = 15, random_state = 0).fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

print("X_train_nmf.shape: ", X_train_nmf.shape)

fig, axes = plt.subplots(3, 5, figsize = (15, 12),
	subplot_kw = {'xticks': (), 'yticks': ()})

for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
	ax.imshow(component.reshape(image_shape))
	ax.set_title("{}. component".format(i))

compn = 8
print(X_train_nmf[:, compn])
fig, axes = plt.subplots(2, 5, figsize = (15, 8),
	subplot_kw = {'xticks': (), 'yticks': ()})
fig.suptitle("Large component 8")
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
	ax.imshow(X_train[ind].reshape(image_shape))

compn = 11
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize = (15, 8),
	subplot_kw = {'xticks': (), 'yticks': ()})
fig.suptitle("Large component 11")
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
	ax.imshow(X_train[ind].reshape(image_shape))

plt.show()
