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
from scipy.integrate import dblquad



jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

"""
All the parameters you want to change are contained in the file config.py

"""

# start the timer

def setup(interaction_type):
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
    system.set_wf(config.wf_type, config.nparticles, config.dim)
    system.set_hamiltonian(type_=config.hamiltonian, int_type=interaction_type, omega=1.0)
    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)
    system.set_optimizer(optimizer=config.optimizer,eta=config.eta)
    cycles , a_values , b_values , W_values , energies = system.train(
        max_iter=config.training_cycles,
        batch_size=config.batch_size,
        seed=config.seed,
    )


    wave_function = system.wf
    return wave_function

wf_non_int = setup("None")
wf_int = setup("Coulomb")

def one_body_density(wf, r_values, n_samples=10000):
    densities = []
    n_particles = 2
    n_dim = 2

    for r in r_values:
        sample_densities = []
        for _ in range(n_samples):
            # Generate random positions for other particles
            other_particles = np.random.randn(n_particles - 1, n_dim)

            # Combine the fixed particle at distance r with other particles
            r_matrix = np.vstack(([r, 0], other_particles))

            # Evaluate the wave function and compute the density
            density = 2* (wf(r_matrix))
            sample_densities.append(np.exp(density))
        
        # Average density over all samples
        avg_density = np.mean(sample_densities)
        densities.append(avg_density)
    
    densities = np.array(densities)
    return densities / np.trapz(densities, r_values)

# Define a range of distances from the origin
r_values = np.linspace(0, 4, 300)  # Adjust the range and number of points as needed

# Calculate the one-body densities for wf_int and wf_non_int
density_int = one_body_density(wf_int, r_values)
density_non_int = one_body_density(wf_non_int, r_values)


np.savetxt(f"data_analysis/r_values.dat", r_values)
np.savetxt(f"data_analysis/density_int.dat", density_int)
np.savetxt(f"data_analysis/density_non_int.dat", density_non_int)








