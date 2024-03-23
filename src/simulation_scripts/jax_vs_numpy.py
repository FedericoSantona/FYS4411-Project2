import sys

"""
with open("../../identity.txt") as file:
    path = file.read()
    path = str(path).strip()

sys.path.append(str(path)) # append yout path to the src folder
"""

sys.path.append("/mnt/c/Users/annar/OneDrive/Desktop/FYS4411/Repo/src")

import jax
import numpy as np
import matplotlib.pyplot as plt


from qs import quantum_state
import config


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

"""
All the parameters you want to change are contained in the file config.py

"""
# set up the system with its backend and level of logging, seed, and other general properties depending on how you want to run it

alpha_values = np.array([ 0.2 , 0.3, 0.4, 0.5, 0.6, 0.7 , 0.8, 0.9, 1.0])
samples = np.array([2**10 ,2**12,  2**14 , 2**16 ])
energies = []
variances = []
error = []

def run_system(alpha_values , n_samples):

    energies = []
    variances = []
    error = []

    for i in alpha_values:
        


        system = quantum_state.QS(
        backend=config.backend,
        log=True,
        logger_level="INFO",
        seed=config.seed,
        alpha=i,
        beta=config.beta,
        radius = config.radius,
        time_step=config.time_step,
        diffusion_coeff=config.diffusion_coeff,
        type_hamiltonian = config.hamiltonian
        )


        # set up the wave function with some of its properties 
        system.set_wf(
            config.wf_type,
            config.nparticles,
            config.dim,
        )


        # choose the hamiltonian
        system.set_hamiltonian(type_=config.hamiltonian, int_type=config.interaction, omega=1.0)


        # choose the sampler algorithm and scale
        system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)


        # choose the optimizer, learning rate, and other properties depending on the optimizer
        system.set_optimizer(
            optimizer=config.optimizer,
            eta=config.eta,
        )

        # train the system, meaning we find the optimal variational parameters for the wave function
        system.train(
            max_iter=config.training_cycles,
            batch_size=config.batch_size,
            seed=config.seed,
        )

        # now we get the results or do whatever we want with them
        results, _, _ = system.sample(n_samples, nchains=config.nchains, seed=config.seed)



        print("alpha", i)
        print("energy", results.energy)
        energies.append(results.energy)
        error.append(results.std_error)
        variances.append(results.variance)

    return np.array(energies), np.array(variances)
   
energy_matrix = np.zeros(( len(samples), len(alpha_values)))

variance_matrix = np.zeros(( len(samples) , len(alpha_values)))

i = 0
                           
for n_samples in samples:

   

    energy, variance = run_system(alpha_values , n_samples)

    energy_matrix[i,:] = np.squeeze(energy)
    variance_matrix[i,:] =np.squeeze( variance)

    i += 1
  

print("Alpha values", alpha_values)
print("Energies", energies)
print("Errors", error)
print("Variances", variances)


fig, ax = plt.subplots(2, figsize=(10, 10))


# Plotting Energy
ax[0].plot(alpha_values, energy_matrix[0], "o-", label=" n_samples = 2**12")
ax[0].plot(alpha_values, energy_matrix[1], "o-", label=" n_samples = 2**14")
ax[0].plot(alpha_values, energy_matrix[2], "o-", label=" n_samples = 2**16")
ax[0].plot(alpha_values, energy_matrix[3], "o-", label=" n_samples = 2**18")
ax[0].set_xlabel("Alpha")
ax[0].set_ylabel("Energy")
ax[0].set_title("Energy as a function of alpha in JAX calculation , N = 10 , dim = 3")
ax[0].legend()

# Plotting Variance
ax[1].plot(alpha_values, variance_matrix[0], "o-", label="n_samples = 2**12")
ax[1].plot(alpha_values, variance_matrix[1], "o-", label=" n_samples = 2**14")
ax[1].plot(alpha_values, variance_matrix[2], "o-", label=" n_samples = 2**16")
ax[1].plot(alpha_values, variance_matrix[3], "o-", label=" n_samples = 2**18")
ax[1].set_xlabel("Alpha")
ax[1].set_ylabel("Variance")
ax[1].set_title("Variance as a function of alpha in JAX calculation , N = 10 , dim = 3")
ax[1].legend()


fig.savefig("jax_vs_numpy.png")

