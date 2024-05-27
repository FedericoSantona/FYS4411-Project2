import sys
import os
import time



# The following will add the path to the ../src directory, for any given laptop running the code
# Assuming the structure of the folders are the same as Daniel initially built (i.e /src is the parent of /simulation script etc.)
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)


import jax


from qs import quantum_state
import config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

"""
All the parameters you want to change are contained in the file config.py

"""
    
eta_values = [0.0001, 0.001, 0.01, 0.1]
n_hidden_values = [2, 4, 6, 8, 10]


energy_values = np.zeros((len(eta_values) * len(n_hidden_values)))
energy_values = energy_values.reshape(len(eta_values), len(n_hidden_values))


for i, eta in enumerate(eta_values):
    for j , n_hidden in enumerate(n_hidden_values):
        
        
        # set up the system with its backend and level of logging, seed, and other general properties depending on how you want to run it
        system = quantum_state.QS(
            backend=config.backend,
            log=True,
            h_number=n_hidden,
            logger_level="INFO",
            seed=config.seed,
            radius = config.radius,
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
        system.set_hamiltonian(type_=config.hamiltonian, int_type=config.interaction, omega=1.0)

        # choose the sampler algorithm and scale
        system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)


        # choose the optimizer, learning rate, and other properties depending on the optimizer
        system.set_optimizer(
            optimizer=config.optimizer,
            eta=eta,
        )

        print("System initialization: Complete..")

        # train the system, meaning we find the optimal variational parameters for the wave function
        cycles , a_values , b_values , W_values , energies = system.train(
            max_iter=config.training_cycles,
            batch_size=config.batch_size,
            seed=config.seed,
        )

        # now we get the results or do whatever we want with them
        results , _ , _  = system.sample(config.nsamples, nchains=config.nchains, seed=config.seed)

        print("Result Energy: ", results.energy)
        print(f"Acceptance rate: {results.accept_rate}")

        energy_values[i , j] = results.energy


np.savetxt(f"data_analysis/eta_values_{config.particle_type}_{config.nparticles}_{config.optimizer}_{config.mcmc_alg}.dat", eta_values)
np.savetxt(f"data_analysis/n_hidden_values_{config.particle_type}_{config.nparticles}_{config.optimizer}_{config.mcmc_alg}.dat", n_hidden_values)
np.savetxt(f"data_analysis/energy_values_{config.particle_type}_{config.nparticles}_{config.optimizer}_{config.mcmc_alg}.dat", energy_values)





