import numpy as np
import matplotlib.pyplot as plt

cycles = np.loadtxt("cycles.dat")
a_values = np.loadtxt("a_values.dat")
b_values = np.loadtxt("b_values.dat")
W_values = np.loadtxt("W_values.dat")
energies = np.loadtxt("energies.dat")

#Plot a values
for i in range(len(a_values.T)):
    plt.plot(cycles, np.transpose(a_values)[i], label=f"a_{i}")

plt.legend()
plt.savefig("a_values.pdf")
plt.close()

#Plot b values
for i in range(len(b_values.T)):
    plt.plot(cycles, np.transpose(b_values)[i], label=f"b_{i}")

plt.legend()
plt.savefig("b_values.pdf")
plt.close()


# Plot W values
for i in range(len(W_values.T)):
    plt.plot(cycles, np.transpose(W_values)[i], label=f"W_{i}")

plt.savefig("W_values.pdf")
plt.close()


#Plot energies
plt.plot(cycles, energies, label="Energy")
plt.legend()
plt.savefig("energy.pdf")