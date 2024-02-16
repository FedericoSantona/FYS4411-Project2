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

alpha_values = np.array([0.2,0.4,0.5,0.6,0.8])
energies = []
variances = []

for i in alpha_values:
    


    system = quantum_state.QS(
    backend=config.backend,
    log=True,
    logger_level="INFO",
    seed=config.seed,
    alpha=i
    )


    # set up the wave function with some of its properties 
    system.set_wf(
        config.wf_type,
        config.nparticles,
        config.dim,
    )


    # choose the sampler algorithm and scale
    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)
    

    # choose the hamiltonian
    system.set_hamiltonian(type_="ho", int_type="Coulomb", omega=1.0)


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
    variances.append(results.variance)



print("Energies", energies)
print("Variances", variances)

# Plotting Energy
plt.plot(alpha_values, energies, "o-", label="Energy")
plt.xlabel("Alpha")
plt.ylabel("Energy")
plt.title("Energy as a function of alpha")
plt.legend()
plt.show()


# Plotting Variance
plt.plot(alpha_values, variances, "o-", label="Variance")
plt.xlabel("Alpha")
plt.ylabel("Variance")
plt.title("Variance as a function of alpha")
plt.legend()
plt.show()

