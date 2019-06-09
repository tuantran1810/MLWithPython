from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF, PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn
import numpy as np

splitStr = "\n" + "=" * 100 + "\n"

S = mglearn.datasets.make_signals()
plt.figure(figsize = (6, 1))
plt.plot(S, '-')
plt.xlabel("Time")
plt.ylabel("Signal")

A = np.random.RandomState(0).uniform(size = (100, 3))
X = np.dot(S, A.T)
print(splitStr)
print("shape of measurement: {}".format(X.shape))
print("shape of signal: {}".format(S.shape))

nmf = NMF(n_components = 3, random_state = 42)
S_ = nmf.fit_transform(X)

pca = PCA(n_components = 3)
H = pca.fit_transform(X)

print("Recovered signal shape: {}".format(S_.shape))

models = [X, S, S_, H]
names = ["Observation", "True source", "NMF Recovered", "PCA Recovered"]

fig, axes = plt.subplots(4, figsize = (8, 4), gridspec_kw = {'hspace': 0.5},
	subplot_kw = {'xticks': (), 'yticks': ()})
for model, name, ax in zip(models, names, axes):
	ax.set_title(name)
	ax.plot(model[:, :3], '-')
plt.show()
