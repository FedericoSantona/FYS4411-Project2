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

alpha_values = np.array([ 0.1 ,0.2, 0.3 , 0.4,0.5, 0.6, 0.7 , 0.8, 0.9, 1 ,1.1 , 1.2 ,1.3, 1.4 ,1.5, 1.6])
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
    time_step=config.time_step,
    diffusion_coeff=config.diffusion_coeff
    )


    # set up the wave function with some of its properties 
    system.set_wf(
        config.wf_type,
        config.nparticles,
        config.dim,
    )


    # choose the hamiltonian
    system.set_hamiltonian(type_=config.hamiltonian, int_type="Coulomb", omega=1.0)


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
    results, _, _ = system.sample(config.nsamples, nchains=config.nchains, seed=config.seed)

    print("alpha", i)
    print("energy", results.energy)
    energies.append(results.energy)
    error.append(results.std_error)
    variances.append(results.variance)


print("Alpha values", alpha_values)
print("Energies", energies)
print("Errors", error)
print("Variances", variances)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Plotting Energy
ax[0].plot(alpha_values, energies, "o-", label="Energy")
ax[0].set_xlabel("Alpha")
ax[0].set_ylabel("Energy")
ax[0].set_title("Energy as a function of alpha")
ax[0].legend()

# Plotting Variance
ax[1].plot(alpha_values, variances, "o-", label="Variance")
ax[1].set_xlabel("Alpha")
ax[1].set_ylabel("Variance")
ax[1].set_title("Variance as a function of alpha")
ax[1].legend()


fig.savefig("energy_alpha.png")

