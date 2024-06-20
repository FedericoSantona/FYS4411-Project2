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
    system.set_hamiltonian(type_=config.hamiltonian, int_type=config.interaction, omega=1.0)
    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)
    system.set_optimizer(optimizer=config.optimizer,eta=config.eta)
    cycles , a_values , b_values , W_values , energies = system.train(
        max_iter=config.training_cycles,
        batch_size=config.batch_size,
        seed=config.seed,
    )


    wave_function = system.wf
    return wave_function

wave_function_non_int = setup("None")
wave_function_int = setup("Coulomb")

# Parameters
x_min, x_max = -5.0, 5.0
y_min, y_max = -5.0, 5.0
N = 100  # Number of points for numerical integration

# Grid for (x1, y1)
x1 = np.linspace(x_min, x_max, N)
y1 = np.linspace(y_min, y_max, N)
X1, Y1 = np.meshgrid(x1, y1)

# Initialize the density array
rho_non_int = np.zeros_like(X1)
rho_int = np.zeros_like(X1)
r = np.sqrt(X1**2 + Y1**2)

# Calculate the one-body density by integrating out (x2, y2)
for i in range(N):
    for j in range(N):
        def integrand_non_int(x2, y2):
            r = np.array([[X1[i, j], Y1[i, j]], [x2, y2]])
            return np.abs(wave_function_non_int(r))**2 
        def integrand_int(x2, y2):
            r = np.array([[X1[i, j], Y1[i, j]], [x2, y2]])
            return np.abs(wave_function_int(r))**2
        
        result_non_int, _ = dblquad(integrand_non_int, x_min, x_max, lambda x2: y_min, lambda x2: y_max)
        result_int, _ = dblquad(integrand_int, x_min, x_max, lambda x2: y_min, lambda x2: y_max)
        rho_non_int[i, j] = result_non_int
        rho_int[i, j] = result_int

# Normalize the density
dx = (x_max - x_min) / (N - 1)
dy = (y_max - y_min) / (N - 1)
rho_non_int /= np.trapz(np.trapz(rho_non_int, dx=dx), dx=dy)
rho_int /= np.trapz(np.trapz(rho_int, dx=dx), dx=dy)


# Bin the results by radial distance
r_flat = r.flatten()
rho_flat_non_int = rho_non_int.flatten()
rho_flat_int = rho_int.flatten()
r1 = np.linspace(0, np.max(r_flat), N)
rho1_non_int = np.zeros_like(r1)
rho1_int = np.zeros_like(r1)

for i in range(len(r1)-1):
    mask = (r_flat >= r1[i]) & (r_flat < r1[i+1])
    if np.sum(mask) > 0:
        rho1_non_int[i] = np.mean(rho_flat_non_int[mask])
        rho1_int[i] = np.mean(rho_flat_int[mask])



np.savetxt(f"data_analysis/X1.dat", X1)
np.savetxt(f"data_analysis/Y1.dat", Y1)
np.savetxt(f"data_analysis/r1.dat", r1)
np.savetxt(f"data_analysis/rho_non_int.dat", rho_non_int)
np.savetxt(f"data_analysis/rho_int.dat", rho_int)
np.savetxt(f"data_analysis/rho1_non_int.dat", rho1_non_int)
np.savetxt(f"data_analysis/rho1_int.dat", rho1_int)








