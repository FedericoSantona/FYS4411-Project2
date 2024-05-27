import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import config

# data for training
def training_plot(particle_type, nparticles):

    cycles = np.loadtxt(f"data_analysis/cycles_{particle_type}_{nparticles}.dat")
    a_values = np.loadtxt(f"data_analysis/a_values_{particle_type}_{nparticles}.dat")
    b_values = np.loadtxt(f"data_analysis/b_values_{particle_type}_{nparticles}.dat")
    W_values = np.loadtxt(f"data_analysis/W_values_{particle_type}_{nparticles}.dat")
    energies = np.loadtxt(f"data_analysis/energies_{particle_type}_{nparticles}.dat")

    sns.set(style="whitegrid")
    if a_values.ndim ==1:
        plt.plot(cycles, a_values, label=f"a")
    else: 
        # Plot a_values
        for i in range(a_values.shape[1]):
            plt.plot(cycles, a_values[:, i], label=f"a_{i}")
    #plt.title(f"A Values Over Cycles of {particle_type} with {nparticles} particles")
    plt.xlabel("Cycles")
    plt.ylabel("a Values")
    plt.legend()
    plt.savefig("figures/a_values.pdf")
    plt.close()

    # Plot b_values
    for i in range(b_values.shape[1]):
        plt.plot(cycles, b_values[:, i], label=f"b_{i}")
    #plt.title(f"B Values Over Cycles of {particle_type} with {nparticles} particles")
    plt.xlabel("Cycles")
    plt.ylabel("b Values")
    plt.legend()
    plt.savefig("figures/b_values.pdf")
    plt.close()

    # Plot W_values
    for i in range(W_values.shape[1]):
        plt.plot(cycles, W_values[:, i], label=f"W_{i}")
    #plt.title(f"W Values Over Cycles of {particle_type} with {nparticles} particles")
    plt.xlabel("Cycles")
    plt.ylabel("w Values")
    plt.legend()
    plt.savefig("figures/W_values.pdf")
    plt.close()

    # Plot energies
    plt.plot(cycles, energies, label="Energy")
    #plt.title(f"Energy Over Cycles of {particle_type} with {nparticles} particles" )
    plt.xlabel("Cycles")
    plt.ylabel("Energy [a.u.]")
    plt.legend()
    plt.savefig("figures/energy.pdf")
    plt.close()


def bootstrap_plots( particle_type , nparticles):

    n_boot_values = np.loadtxt(f"data_analysis/n_boot_values_{particle_type}_{nparticles}.dat")
    variances_bo = np.loadtxt(f"data_analysis/variances_bo_{particle_type}_{nparticles}.dat")
    variances_boot = np.loadtxt(f"data_analysis/variances_boot_{particle_type}_{nparticles}.dat")
    block_sizes = np.loadtxt(f"data_analysis/block_sizes_{particle_type}_{nparticles}.dat")
    variances_bl = np.loadtxt(f"data_analysis/variances_bl_{particle_type}_{nparticles}.dat")
    variances_block = np.loadtxt(f"data_analysis/variances_block_{particle_type}_{nparticles}.dat")


    # Set Seaborn style
    sns.set(style="whitegrid", context="talk", palette="colorblind")

    # Create a figure with subplots
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    # Plot 1: Bootstrapped Variance vs Normal Variance on the first subplot
    ax[0].plot(n_boot_values, variances_bo, label="Normal Variance", marker='o', linestyle='-', linewidth=2)
    ax[0].plot(n_boot_values, variances_boot, label="Bootstrapped Variance", marker='x', linestyle='--', linewidth=2)
    ax[0].set_xlabel("Number of Bootstraps")
    ax[0].set_ylabel("Variance")
    ax[0].set_title(f"Variance Comparison: Bootstrapped vs Normal of {particle_type} with {nparticles} particles")
    ax[0].legend()

    # Plot 2: Blocking Variance vs Normal Variance on the second subplot
    ax[1].plot(block_sizes, variances_bl, label="Normal Variance", marker='o', linestyle='-', linewidth=2)
    ax[1].plot(block_sizes, variances_block, label="Blocking Variance", marker='x', linestyle='--', linewidth=2)
    ax[1].set_xlabel("Block Size")
    ax[1].set_ylabel("Variance")
    ax[1].set_title(f"Variance Comparison: Blocking vs Normal of {particle_type} with {nparticles} particles")
    ax[1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig("figures/variance_comparisons.pdf")
    plt.close()

    # Compare bootstrapping and blocking
    plt.figure(figsize=(12, 8))
    plt.plot(n_boot_values, variances_boot, label="Variance with bootstrap", marker='o', linestyle='-', linewidth=2)
    plt.plot(block_sizes, variances_block, label="Variance with Blocking", marker='x', linestyle='--', linewidth=2)
    plt.xlabel("Number of Bootstraps / Block Sizes")
    plt.ylabel("Variance")
    plt.title(f"Variance Comparison: Bootstrapping vs Blocking of {particle_type} with {nparticles} particles")
    plt.legend()
    plt.savefig("figures/boot_vs_blocking.pdf")


def plot_int_energy_vs_particles():

    energies_bosons = np.loadtxt("data_analysis/int_bosons_energies.dat")
    energies_fermions = np.loadtxt("data_analysis/int_fermion_energies.dat")
    n_particles = np.loadtxt("data_analysis/n_particles.dat")


    # Set Seaborn style
    sns.set(style="whitegrid", palette="muted")

    # Create the plot
    plt.figure(figsize=(10, 6))  # Optionally increase figure size for better readability
    plt.plot(n_particles, energies_fermions, 'o-', label="Fermions", linewidth=2, markersize=8)
    plt.plot(n_particles, energies_bosons, 's-', label="Bosons", linewidth=2, markersize=8)
    plt.xlabel("Number of particles")
    plt.ylabel("Interacting Energy [a.u.]")
    plt.title("Energy vs Number of Particles")
    plt.legend(title="Particle Type")
    plt.grid(True)  # Ensure the grid is enabled

    # Save the figure
    plt.savefig("figures/int_energy_vs_particles.pdf")


def plot_energy_vs_particles():

    energies_bosons = np.loadtxt("data_analysis/bosons_energies.dat")
    energies_fermions = np.loadtxt("data_analysis/fermion_energies.dat")
    n_particles = np.loadtxt("data_analysis/n_particles.dat")

    en_fermions_teo =  np.array([1 , 4 , 9 , 16  , 25 ])
    en_bosons_teo =  np.array([1 , 2 , 3 , 4 , 5 ])


    # Set Seaborn style
    sns.set(style="whitegrid", palette="muted")

    # Create the plot
    plt.figure(figsize=(10, 6))  # Optionally increase figure size for better readability
    plt.plot(n_particles, energies_fermions, 'o-', label="Fermions", linewidth=2, markersize=8, color="blue")
    plt.plot(n_particles, energies_bosons, 's-', label="Bosons", linewidth=2, markersize=8 , color="green")
    plt.plot(n_particles, en_bosons_teo, '--', label="theoretical", linewidth=2, markersize=8 , color="red")
    plt.plot(n_particles, en_fermions_teo, '--', linewidth=2, markersize=8 , color="red")
    plt.xlabel("Number of particles")
    plt.ylabel("Energy [a.u.]")
    plt.title("Energy vs Number of Particles")
    plt.legend(title="Particle Type")
    plt.grid(True)  # Ensure the grid is enabled

    # Save the figure
    plt.savefig("figures/energy_vs_particles.pdf")


def plot_heatmap(part_typ, n_part, optimizer, mcmc_alg):
    

    eta_values = np.loadtxt(f"data_analysis/eta_values_{part_typ}_{n_part}_{optimizer}_{mcmc_alg}.dat")
    n_hidden_values = np.loadtxt(f"data_analysis/n_hidden_values_{part_typ}_{n_part}_{optimizer}_{mcmc_alg}.dat")
    energy_values = np.loadtxt(f"data_analysis/energy_values_{part_typ}_{n_part}_{optimizer}_{mcmc_alg}.dat")


    # Create the heatmap using seaborn
    
    data = pd.DataFrame(energy_values,index = eta_values, columns = n_hidden_values)
    #breakpoint()
    ax = sns.heatmap(data, annot=True, fmt=".2f")
    #sns.heatmap(data, annot=True, fmt=".2f")

    # Invert the y-axis to have the (0,0) on the top left corner
    ax.invert_yaxis()

    plt.xlabel("Number of hidden units")
    plt.ylabel("Learning rate")
    #plt.title("Energy Values Heatmap")

    # Save the figure
    plt.savefig(f"figures/grid_search_{part_typ}_{n_part}_{optimizer}_{mcmc_alg}.pdf")
    # If you want to also display the heatmap


def position_plot(nparticles,  particle_type , nsamples):

    sampled_positions = np.loadtxt(f"data_analysis/sampled_positions_{particle_type}_{nparticles}_{nsamples}.dat")


   
    # Assuming each column is a particle and each row is a sample, create a DataFrame
    df = pd.DataFrame(sampled_positions, columns=[f'Particle {i+1}' for i in range(sampled_positions.shape[1])])
    df['Sample'] = df.index % 100  # Recreate the sample index if needed, modify '100' based on actual samples per chain

    # Melting the DataFrame to use Seaborn easily
    df_melted = df.melt(id_vars=['Sample'], var_name='Particle', value_name='Distance')

    # Set up the FacetGrid
    g = sns.FacetGrid(df_melted, col="Particle", col_wrap=3, height=4, aspect=1.5, hue='Sample', palette='viridis')
    g.map(plt.scatter, 'Sample', 'Distance', alpha=0.6, s=20)  # Scatter plot for each particle's position over time

    # Adding a line plot to connect the positions, to show the movement over samples
    g.map(plt.plot, 'Sample', 'Distance', alpha=0.3)

    # Enhance the plot
    g.set_titles("{col_name}")
    g.add_legend(title="Sample Index")

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Particle Distances Over Time for {nparticles} {particle_type}  with {nsamples} samples ')  # Overall title
    
    # Save the figure
    plt.savefig("figures/position_plot.pdf")


def print_table():
    
    #load the data
    energies_bosons = np.loadtxt("data_analysis/bosons_energies.dat")
    energies_fermions = np.loadtxt("data_analysis/fermion_energies.dat")
    n_particles = np.loadtxt("data_analysis/n_particles.dat")
    variances_bosons = np.loadtxt("data_analysis/bosons_variances.dat")
    variances_fermions = np.loadtxt("data_analysis/fermion_variances.dat")  

    #create the table

    # Creating a dictionary with column names and lists
    data = {
        'n_particles': n_particles,
        'energies_bosons': energies_bosons,
        'energies_fermions': energies_fermions,
        'variances_bosons': variances_bosons,
        'variances_fermions': variances_fermions
    }

    # Converting the dictionary into a DataFrame
    df = pd.DataFrame(data)

    # Printing the DataFrame
    print(df)



training_plot(config.particle_type, config.nparticles)
#bootstrap_plots(config.particle_type, config.nparticles)
#plot_int_energy_vs_particles()
#plot_energy_vs_particles()
#plot_heatmap("bosons","1","adam","m")
#plot_heatmap("bosons","2","adam","mh")
#position_plot(config.nparticles, config.particle_type, config.nsamples)






    



