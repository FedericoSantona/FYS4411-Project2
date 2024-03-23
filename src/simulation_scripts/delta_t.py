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

delta_t_values = np.array([1e-8 , 1e-5 , 1e-3 ,  1 , 1e3, 1e5 , 1e7])
alpha_values = np.array([ 0.2 , 0.3, 0.4, 0.5, 0.6,  0.8 , 1.0])
energy_ana = np.zeros((len(alpha_values), len(delta_t_values)))
energy_jax = np.zeros((len(alpha_values), len(delta_t_values)))

def setup_and_train(backend, delta_t , alpha_value = config.alpha):
    
    system = quantum_state.QS(
        backend=backend,
        log=True,
        logger_level="INFO",
        seed=config.seed,
        alpha=alpha_value,
        beta=config.beta,
        time_step=delta_t,
        diffusion_coeff=config.diffusion_coeff
    )
    
    # Adjust parameters based on sample_size if necessary
    
    # Setup and training process
    system.set_wf(config.wf_type, config.nparticles, config.dim)
    system.set_hamiltonian(type_=config.hamiltonian, int_type=config.interaction, omega=1.0)
    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)
    system.set_optimizer(optimizer=config.optimizer, eta=config.eta)
    system.train(max_iter=config.training_cycles, batch_size=config.batch_size, seed=config.seed)

    results, _, _ = system.sample(config.nsamples, nchains=config.nchains, seed=config.seed)

    
    
    return results.energy

for i, delta_t in enumerate(delta_t_values):
    for j, alpha in enumerate(alpha_values):
        # You might need to adjust the call to setup_and_train based on whether it's 'ana' or 'jax' backend
        energy_ana[j, i] = setup_and_train('numpy', delta_t, alpha)
        energy_jax[j, i] = setup_and_train('jax', delta_t, alpha)



# Assuming your energy_ana and energy_jax matrices are populated as described above
fig, ax = plt.subplots(1, 2, figsize=(14, 6))  # Two plots side by side

# Setting the extent for imshow, ensuring the x-axis values (delta_t) are represented in log scale
extent = [np.log10(delta_t_values.min()), np.log10(delta_t_values.max()), alpha_values.min(), alpha_values.max()]

# Heatmap for analytical backend
pos = ax[0].imshow(energy_ana, cmap='viridis', aspect='auto', origin='lower', extent=extent)
ax[0].set_title(f'Energy (Ana) vs. Log(Delta_t) and Alpha')
ax[0].set_xlabel('Log(Delta_t)')
ax[0].set_ylabel('Alpha')
fig.colorbar(pos, ax=ax[0])

# Setting the x-axis to show the actual log10 values of delta_t
ax[0].set_xticks(np.log10(delta_t_values))
ax[0].set_xticklabels([f'{val:f}' for val in delta_t_values])

# Heatmap for JAX backend
pos = ax[1].imshow(energy_jax, cmap='viridis', aspect='auto', origin='lower', extent=extent)
ax[1].set_title(f'Energy (Jax) vs. Log(Delta_t) and Alpha')
ax[1].set_xlabel('Log(Delta_t)')
# We only set the ylabel for the first subplot

# Setting the x-axis to show the actual log10 values of delta_t for the second plot as well
ax[1].set_xticks(np.log10(delta_t_values))
ax[1].set_xticklabels([f'{val:f}' for val in delta_t_values])

fig.colorbar(pos, ax=ax[1])
plt.tight_layout()
plt.savefig("delta_t_heatmaps.png")