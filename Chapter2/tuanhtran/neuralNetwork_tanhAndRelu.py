import numpy as np
import matplotlib.pyplot as plt

line = np.linspace(-3, 3, 100)

plt.plot(line, np.tanh(line), label = "tanh")
plt.plot(line, np.maximum(line, 0), label = "relu")

plt.legend(loc = "best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")
plt.show()
