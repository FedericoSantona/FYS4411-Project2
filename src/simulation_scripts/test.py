import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
x_values = np.linspace(0, 10, 100)
y_values = np.sin(x_values)

plt.plot(x_values, y_values)
plt.show()
