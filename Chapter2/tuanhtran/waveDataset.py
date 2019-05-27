import matplotlib.pyplot as plt
import mglearn

x, y = mglearn.datasets.make_wave(n_samples = 40)
plt.plot(x, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()
