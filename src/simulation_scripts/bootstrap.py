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

n_boot_values = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 ,2048 , 4096])
block_sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 ,2048 , 4096])

variances_bo = []
variances_bl = []
variances_boot = []
variances_block = []

    
    
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

# Adjust parameters based on sample_size if necessary

# Setup and training process
system.set_wf(config.wf_type, config.nparticles, config.dim)
system.set_hamiltonian(type_=config.hamiltonian, int_type=config.interaction, omega=1.0)
system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)
system.set_optimizer(optimizer=config.optimizer, eta=config.eta)
system.train(max_iter=config.training_cycles, batch_size=config.batch_size, seed=config.seed)

results, sampled_positions, local_energies = system.sample(config.nsamples, nchains=config.nchains, seed=config.seed)

def stat_analysis(n_bootstraps, block_size):

    mean_energy_boot, variance_energy_boot = system.superBoot(local_energies, n_bootstraps)

    block_variance = system.blocking_method(local_energies , block_size )
    
    
    return  results.variance, variance_energy_boot , block_variance


for n_bootstrap in n_boot_values:
    variance, variance_boot , _ = stat_analysis(n_bootstrap, 1)
    
    variances_bo.append(variance)
    variances_boot.append(variance_boot)

for block_size in block_sizes:
    variance, _ ,  block_variance = stat_analysis(1, block_size)
    
    variances_bl.append(variance)
    variances_block.append(block_variance)



np.savetxt(f"data_analysis/n_boot_values_{config.particle_type}_{config.nparticles}.dat", n_boot_values)
np.savetxt(f"data_analysis/variances_bo_{config.particle_type}_{config.nparticles}.dat", variances_bo)
np.savetxt(f"data_analysis/variances_boot_{config.particle_type}_{config.nparticles}.dat", variances_boot)
np.savetxt(f"data_analysis/block_sizes_{config.particle_type}_{config.nparticles}.dat", block_sizes)
np.savetxt(f"data_analysis/variances_bl_{config.particle_type}_{config.nparticles}.dat", variances_bl)
np.savetxt(f"data_analysis/variances_block_{config.particle_type}_{config.nparticles}.dat", variances_block)


