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

delta_t_values = np.array([1e-5 , 1e-4 , 1e-3 , 1e-2 , 1e-1 , 1 , 10, 100 , 1000])
energy_ana = []
energy_jax = []

def setup_and_train(backend, delta_t):
    
    system = quantum_state.QS(
        backend=backend,
        log=True,
        logger_level="INFO",
        seed=config.seed,
        alpha=config.alpha,
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

for delta_t in delta_t_values:
    energy_ana.append(setup_and_train("numpy", delta_t))
    energy_jax.append(setup_and_train("jax", delta_t))

print("Execution energy for numpy:", energy_ana)
print("Execution energy for jax:", energy_jax)


fig, ax = plt.subplots(2, 1, figsize=(10, 10))  # Adjust for a vertical stack of two plots

# First plot: Execution energy for Numpy and Jax
ax[0].plot(np.log(delta_t_values), energy_ana, label="Numpy")
ax[0].set_xlabel("Log10(delta_t)")
ax[0].set_ylabel("Energy")
ax[0].set_title("Energy in dependance of delta_t for Numpy alpha = 0.5")
ax[0].legend()

# Second plot: Difference in execution energy
ax[1].plot(np.log(delta_t_values), energy_jax, label="Jax")
ax[1].set_xlabel("Log10(delta_t)")
ax[1].set_ylabel("Energy")
ax[1].set_title("Energy in dependance of delta_t for Jax alpha = 0.5")
ax[1].legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig("delta_t.png")  # Save the figure as a single image