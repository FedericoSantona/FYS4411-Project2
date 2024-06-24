# import all function from module.NaiveNN
import numpy as np
import matplotlib.pyplot as plt

from module.NaiveNN import *


# define the N, D of the system
N = 2
D = 2
H = 10

# initialize the parameters
RW_params = init_params_guess(N, D, H, alpha=-0.5, epsilon=0.5, gamma=10.0)


# get the mean energy with optimized params

run_step = 100000
RW_chain = metropolis(N, D, RW_params, run_step, 0.1)

# get the energy
RW_energy=0
for i in range(run_step):
    RW_energy += local_potential_energy(RW_chain[i]) + local_kinetic_energy(RW_chain[i], RW_params)
RW_energy /= run_step

print("The mean energy RW: ", RW_energy)

num_iterations = 1000
# optimize
GD_opt_params, GD_energies = optimization(N, D, H, RW_params,
                                          optimization_steps=num_iterations, batch_size=500, lr=0.01, decay=0.9, verbose=False)

# optimize with adam
ADAM_opt_params, ADAM_energies = optimization_adam(N, D, H, RW_params,
                                                   optimization_steps=num_iterations, batch_size=500, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, verbose=False)



print("The energy obtained with ADAM :" ,np.mean( ADAM_energies))
print("The energy obtained  GD : ", np.mean(GD_energies) )


iterations = np.arange(0, num_iterations )



np.savetxt(f"data_analysis/ADAM_energies_NN.dat", ADAM_energies)
np.savetxt(f"data_analysis/GD_energies_NN.dat.dat", GD_energies)
np.savetxt(f"data_analysis/iterations_NN.dat", iterations)







"""
GD_chain = metropolis(N, D, GD_opt_params, run_step, 0.1)
GD_energy=0
for i in range(run_step):
    GD_energy += local_potential_energy(GD_chain[i]) + local_kinetic_energy(GD_chain[i], GD_opt_params)
GD_energy /= run_step

print("The mean energy GD: ", GD_energy)


ADAM_chain = metropolis(N, D, ADAM_opt_params, run_step, 0.1)
ADAM_energy=0
for i in range(run_step):
    ADAM_energy += local_potential_energy(ADAM_chain[i]) + local_kinetic_energy(ADAM_chain[i], ADAM_opt_params)
ADAM_energy /= run_step

print("The mean energy ADAM: ", ADAM_energy)

"""