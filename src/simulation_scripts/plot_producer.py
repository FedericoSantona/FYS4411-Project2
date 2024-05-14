import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# data for training
def training_plot() :
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


def bootstrap_plots():

    n_boot_values = np.loadtxt("n_boot_values.dat")
    variances_bo = np.loadtxt("variances_bo.dat")
    variances_boot = np.loadtxt("variances_boot.dat")
    block_sizes = np.loadtxt("block_sizes.dat")
    variances_bl = np.loadtxt("variances_bl.dat")
    variances_block = np.loadtxt("variances_block.dat")


    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    # Plot 1: Bootstrapped Variance vs Normal Variance on the first subplot
    ax[0].plot(n_boot_values, variances_bo, label="Normal Variance", marker='o', linestyle='-')
    ax[0].plot(n_boot_values, variances_boot, label="Bootstrapped Variance", marker='x', linestyle='--')
    ax[0].set_xlabel("Number of Bootstraps")
    ax[0].set_ylabel("Variance")
    ax[0].set_title("Variance Comparison: Bootstrapped vs Normal")
    ax[0].legend()

    # Plot 2: Blocking Variance vs Normal Variance on the second subplot
    ax[1].plot(block_sizes, variances_bl, label="Normal Variance", marker='o', linestyle='-')
    ax[1].plot(block_sizes, variances_block, label="Blocking Variance", marker='x', linestyle='--')
    ax[1].set_xlabel("Block Size")
    ax[1].set_ylabel("Variance")
    ax[1].set_title("Variance Comparison: Blocking vs Normal")
    ax[1].legend()

    # Adjust layout to make room for the titles and labels
    plt.tight_layout()

    # Save the figure as a single image
    plt.savefig("variance_comparisons.png")
    plt.show()

    # clean the figure and start new plot
    plt.clf()

    #change the fig size for the plot

    #COmpare bootstrapping and blocking
    plt.figure(figsize=(12,8))
    plt.plot(n_boot_values, variances_boot ,  label="Variance with bootstrap", marker='o', linestyle='-')
    plt.plot(n_boot_values, variances_block ,  label="Variance with Blocking", marker='x', linestyle='--')
    plt.xlabel("Number of Bootstraps  / Block sizes")
    plt.ylabel("Variance")

    plt.title("Variance Comparison: Bootstrapping vs Blocking")
    plt.legend()
    plt.savefig("boot_vs_blocking.png")
    plt.show()



def plot_energy_vs_particles():

    energies_bosons = np.loadtxt("bosons_energies.dat")
    energies_fermions = np.loadtxt("fermion_energies.dat")
    n_particles = np.loadtxt("n_particles.dat")


    plt.plot(n_particles, energies_fermions, 'o-')
    plt.plot(n_particles, energies_bosons, 'o-')
    plt.xlabel("Number of particles")
    plt.ylabel("Energy")
    plt.legend(["Fermions", "Bosons"])
    plt.savefig("energy_vs_particles.pdf")


def plot_heatmap():

    eta_values = np.loadtxt("eta_values.dat")
    n_hidden_values = np.loadtxt("n_hidden_values.dat")
    energy_values = np.loadtxt("energy_values.dat")


    # Create the heatmap using seaborn
    ax = sns.heatmap(energy_values, annot=True, fmt=".2f", cmap='viridis',
                    xticklabels=n_hidden_values, yticklabels=eta_values)

    # Invert the y-axis to have the (0,0) on the top left corner
    ax.invert_yaxis()

    plt.xlabel("Number of hidden units")
    plt.ylabel("Learning rate")
    plt.title("Energy Values Heatmap")

    # Save the figure
    plt.savefig("grid_search.pdf")
    # If you want to also display the heatmap

    



    




    



