import sys
import os




# The following will add the path to the ../src directory, for any given laptop running the code
# Assuming the structure of the folders are the same as Daniel initially built (i.e /src is the parent of /simulation script etc.)
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)




import jax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from qs import quantum_state
import config


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

"""
All the parameters you want to change are contained in the file config.py

"""
# set up the system with its backend and level of logging, seed, and other general properties depending on how you want to run it




def setup_and_train(particle_type):
    
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
        type_particle = particle_type
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
    alphas ,cycles, _, _, _ = system.train(
        max_iter=config.training_cycles,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    

    # now we get the results or do whatever we want with them
    _ , sampled_position, _ = system.sample(config.nsamples, nchains=config.nchains, seed=config.seed)
    

    return np.array( sampled_position)
    

positions_bos = setup_and_train("bosons")
positions_fer = setup_and_train("fermions")


positions_bos = positions_bos.reshape(-1, config.nparticles, config.dim)
positions_fer = positions_fer.reshape(-1, config.nparticles, config.dim)


                                     
#breakpoint()
# Function to compute relative distances for all samples
# Function to compute relative distances for all samples
def compute_relative_distances(positions):
    all_distances = []
    for i in range(positions.shape[0]):
        sample = positions[i]

        distances = pdist(sample)
        all_distances.extend(distances)
    return np.array(all_distances)

# Compute relative distances for fermions and bosons
relative_distances_fer = compute_relative_distances(positions_fer)
relative_distances_bos = compute_relative_distances(positions_bos)


np.savetxt("data_analysis/rel_dist_bos.dat", relative_distances_bos)
np.savetxt("data_analysis/rel_dist_fer.dat", relative_distances_fer)



