# import all function from module.NaiveNN
import numpy as np
import matplotlib.pyplot as plt

from module.NaiveNN import *

# define the N, D of the system
N = 4
D = 3

# define numnber of hidden lyer
H = 2

params = init_params_guess(N, D, H,
                           alpha=-0.7, epsilon=0.1, gamma=0.1)

# optimize
opt_params = optimization(nn,params, N, D, H, nn_grad_params,
                          local_kinetic_energy, local_potential_energy, 
                          optimization_steps=10, batch_size=500, lr=0.1)

opt_params


# get the mean energy with optimized params

run_step = 10000
chain = metropolis(N, D, nn, opt_params, run_step, 0.1)

# get the energy
energy=0
for i in range(run_step):
    energy += local_potential_energy(chain[i]) + local_kinetic_energy(chain[i], opt_params)
energy /= run_step

print("The mean energy is: ", energy)
print("The analytical energy is: ", 0.5*N*D)