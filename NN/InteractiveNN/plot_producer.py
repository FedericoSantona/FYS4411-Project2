import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plot_energy_vs_iteration():

    iterations = np.loadtxt("data_analysis/iterations_NN.dat")
    ADAM_energies = np.loadtxt("data_analysis/ADAM_energies_NN.dat") 
    GD_energies = np.loadtxt("data_analysis/GD_energies_NN.dat.dat")


    

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=iterations, y=GD_energies, label='GD', marker='o')
    sns.lineplot(x=iterations, y=ADAM_energies, label='ADAM', marker='o')

    # Add titles and labels
    plt.title('Energy Optimization Comparison')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/NN_energy_vs_ite.pdf")


plot_energy_vs_iteration()
