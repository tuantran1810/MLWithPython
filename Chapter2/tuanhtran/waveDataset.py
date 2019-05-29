import matplotlib.pyplot as plt
import mglearn

x, y = mglearn.datasets.make_wave(n_samples = 40)
plt.plot(x, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
mglearn.plots.plot_knn_regression(n_neighbors = 1)
mglearn.plots.plot_knn_regression(n_neighbors = 3)
plt.show()
