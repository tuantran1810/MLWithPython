from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mglearn
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"

people = fetch_lfw_people(min_faces_per_person = 20, resize = 0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize = (15, 8),
	subplot_kw = {'xticks' : (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
	ax.imshow(image)
	ax.set_title(people.target_names[target])

print(splitStr)
print("people.images.shape: {}".format(people.images.shape))
print("number of classes: {}".format(len(people.target_names)))

print(splitStr)
counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
	print("{0:25} {1:3}".format(name, count, end = '	'))
plt.show()

print(splitStr)
mask = np.zeros(people.target.shape, dtype = np.bool)

