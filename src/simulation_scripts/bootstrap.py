import sys
import jax
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append("/mnt/c/Users/annar/OneDrive/Desktop/FYS4411/Repo/src")

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

def setup_and_train(n_bootstraps, block_size):
    start_time = time.perf_counter()
    
    system = quantum_state.QS(
        backend=config.backend,
        log=True,
        logger_level="INFO",
        seed=config.seed,
        alpha=config.alpha,
        beta=config.beta,
        time_step=config.time_step,
        diffusion_coeff=config.diffusion_coeff
    )
    
    # Adjust parameters based on sample_size if necessary
    
    # Setup and training process
    system.set_wf(config.wf_type, config.nparticles, config.dim)
    system.set_hamiltonian(type_=config.hamiltonian, int_type=config.interaction, omega=1.0)
    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)
    system.set_optimizer(optimizer=config.optimizer, eta=config.eta)
    system.train(max_iter=config.training_cycles, batch_size=config.batch_size, seed=config.seed)

    results, sampled_positions, local_energies = system.sample(config.nsamples, nchains=config.nchains, seed=config.seed)

    mean_energy_boot, variance_energy_boot = system.superBoot(local_energies, n_bootstraps)

    block_variance = system.blocking_method(local_energies , block_size )
    
    
    return  results.variance, variance_energy_boot , block_variance


for n_bootstrap in n_boot_values:
    variance, variance_boot , _ = setup_and_train(n_bootstrap, 1)
    
    variances_bo.append(variance)
    variances_boot.append(variance_boot)

for block_size in block_sizes:
    variance, _ ,  block_variance = setup_and_train(1, block_size)
    
    variances_bl.append(variance)
    variances_block.append(block_variance)





fig, ax = plt.subplots(1, 2, figsize=(20, 6))

# Plot 1: Bootstrapped Variance vs Normal Variance on the first subplot
ax[0].plot(n_boot_values, variances_bo, label="Normal Variance", marker='o', linestyle='-')
ax[0].plot(n_boot_values, variances_boot, label="Bootstrapped Variance", marker='x', linestyle='--')
ax[0].set_xlabel("Number of Bootstraps")
ax[0].set_ylabel("Variance")
ax[0].set_title("Variance Comparison: Bootstrapped vs Normal")
ax[0].legend()

# Plot 2: Blocking Variance vs Normal Variance on the second subplot
ax[1].plot(block_sizes, variances_bl, label="Normal Variance", marker='o', linestyle='-')
ax[1].plot(block_sizes, variances_block, label="Blocking Variance", marker='x', linestyle='--')
ax[1].set_xlabel("Number of Bootstraps")
ax[1].set_ylabel("Variance")
ax[1].set_title("Variance Comparison: Blocking vs Normal")
ax[1].legend()

# Adjust layout to make room for the titles and labels
plt.tight_layout()

# Save the figure as a single image
plt.savefig("variance_comparisons.png")


#COmpare bootstrapping and blocking
plt.plot(n_boot_values, variances_boot ,  label="Variance with bootstrap", marker='o', linestyle='-')
plt.plot(block_sizes, variances_block ,  label="Variance with Blocking", marker='x', linestyle='--')
plt.xlabel("Number of Bootstraps  / Block sizes")
plt.ylabel("Variance")
plt.title("Variance Comparison: Bootstrapping vs Blocking")
plt.legend()
plt.savefig("variance_comparisons_boot_blocking.png")
