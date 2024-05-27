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

# start the timer
start_time = time.time()

# set up the system with its backend and level of logging, seed, and other general properties depending on how you want to run it
system = quantum_state.QS(
    backend=config.backend,
    log=True,
    h_number=config.n_hidden,
    logger_level="INFO",
    seed=config.seed,
    radius = config.radius,
    time_step=config.time_step,
    diffusion_coeff=config.diffusion_coeff,
    type_particle = config.particle_type,
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

print("System initialization: Complete..")

# train the system, meaning we find the optimal variational parameters for the wave function
cycles , a_values , b_values , W_values , energies = system.train(
    max_iter=config.training_cycles,
    batch_size=config.batch_size,
    seed=config.seed,
)

# now we get the results or do whatever we want with them
results , sampled_positions , local_energies  = system.sample(config.nsamples, nchains=config.nchains, seed=config.seed)

end_time = time.time()
execution_time = end_time - start_time

sampled_positions = np.array(sampled_positions).reshape(-1, config.nparticles, config.dim)
sampled_positions = np.linalg.norm(sampled_positions, axis=2)

block_size = 1000 # this is the block size for the blocking method

block_energy , block_variance = system.blocking_method(local_energies , block_size )



# display the results
print("Metrics: ", results)
print("Result Energy: ", results.energy)
print("Blocking Energy: ", block_energy)
print("variance " , results.variance)
print("blocking variance " , block_variance)
print("standard deviation from mean " , results.std_error)
print(f"Acceptance rate: {results.accept_rate}")

print(f"Execution time: {execution_time} seconds")

np.savetxt(f"data_analysis/a_values_{config.particle_type}_{config.nparticles}.dat", a_values)
np.savetxt(f"data_analysis/b_values_{config.particle_type}_{config.nparticles}.dat", b_values)
np.savetxt(f"data_analysis/W_values_{config.particle_type}_{config.nparticles}.dat", W_values)
np.savetxt(f"data_analysis/energies_{config.particle_type}_{config.nparticles}.dat", energies)
np.savetxt(f"data_analysis/cycles_{config.particle_type}_{config.nparticles}.dat", cycles)
np.savetxt(f"data_analysis/sampled_positions_{config.particle_type}_{config.nparticles}_{config.nsamples}_{config.interaction}.dat", sampled_positions)




