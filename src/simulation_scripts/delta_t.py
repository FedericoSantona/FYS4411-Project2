import sys
import jax
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# The following will add the path to the ../src directory, for any given laptop running the code
# Assuming the structure of the folders are the same as Daniel initially built (i.e /src is the parent of /simulation script etc.)
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from qs import quantum_state
import config

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

delta_t_values = np.array([0.001 , 0.01 , 0.1 , 0.5 ,  1 ])
energy = np.zeros( len(delta_t_values))
variance = np.zeros( len(delta_t_values))


if config.mcmc_alg != "mh":
    raise ValueError("This script only supports Metropolis-Hastings sampler")

def setup_and_train( delta_t ):
    
    system = quantum_state.QS(
        backend=config.backend,
        log=True,
        h_number=config.n_hidden,
        logger_level="INFO",
        seed=config.seed,
        radius = config.radius,
        time_step=delta_t,
        diffusion_coeff=config.diffusion_coeff,
        type_particle = "bosons"
    )
    # Adjust parameters based on sample_size if necessary
    
    # Setup and training process
    system.set_wf(config.wf_type, config.nparticles, config.dim)
    system.set_hamiltonian(type_=config.hamiltonian, int_type=config.interaction, omega=1.0)
    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)
    system.set_optimizer(optimizer=config.optimizer, eta=config.eta)
    system.train(max_iter=config.training_cycles, batch_size=config.batch_size, seed=config.seed)

    results, _, local_energy = system.sample(config.nsamples, nchains=config.nchains, seed=config.seed)

    block_size = 1000 # this is the block size for the blocking method

    energy , variance = system.blocking_method(local_energy , block_size )
    

    return energy, variance

for i, delta_t in enumerate(delta_t_values):
        # You might need to adjust the call to setup_and_train based on whether it's 'ana' or 'jax' backend
        energy[i] , variance[i] = setup_and_train(delta_t)
       



np.savetxt(f"data_analysis/delta_t_values_{config.particle_type}_{config.nparticles}.dat", delta_t_values)
np.savetxt(f"data_analysis/energy_delta_t{config.particle_type}_{config.nparticles}.dat", energy)
np.savetxt(f"data_analysis/variance_delta_t{config.particle_type}_{config.nparticles}.dat", variance)

