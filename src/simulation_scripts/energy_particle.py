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


from qs import quantum_state
import config


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

"""
All the parameters you want to change are contained in the file config.py

"""
# set up the system with its backend and level of logging, seed, and other general properties depending on how you want to run it


n_particles = np.array([ 2 ,4 , 6 , 8 , 10  ])

energies_fermions = []
energies_bosons = []
variances_fermions = []
variances_bosons = []

for i in n_particles:
    
    # set up the system with its backend and level of logging, seed, and other general properties depending on how you want to run it
    system_bos = quantum_state.QS(
        backend=config.backend,
        log=True,
        h_number=config.n_hidden,
        logger_level="INFO",
        seed=config.seed,
        radius = config.radius,
        time_step=config.time_step,
        diffusion_coeff=config.diffusion_coeff,
        type_particle = "bosons"
    )


    system_fer = quantum_state.QS(
        backend=config.backend,
        log=True,
        h_number=config.n_hidden,
        logger_level="INFO",
        seed=config.seed,
        radius = config.radius,
        time_step=config.time_step,
        diffusion_coeff=config.diffusion_coeff,
        type_particle = "fermions"
    )


    # set up the wave function with some of its properties 
    system_bos.set_wf(
        config.wf_type,
        int(i),
        config.dim,
    )

    system_fer.set_wf(
        config.wf_type,
        int(i),
        config.dim,
    )
    

    # choose the hamiltonian
    system_bos.set_hamiltonian(type_=config.hamiltonian, int_type=config.interaction, omega=1.0)

    system_fer.set_hamiltonian(type_=config.hamiltonian, int_type=config.interaction, omega=1.0)

    # choose the sampler algorithm and scale
    system_bos.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)

    system_fer.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)



    # choose the optimizer, learning rate, and other properties depending on the optimizer
    system_bos.set_optimizer(
        optimizer=config.optimizer,
        eta=config.eta,
    )

    system_fer.set_optimizer(
        optimizer=config.optimizer,
        eta=config.eta,
    )

    # train the system, meaning we find the optimal variational parameters for the wave function
    alphas ,cycles, _, _, _ = system_bos.train(
        max_iter=config.training_cycles,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    alphas ,cycles, _, _, _ = system_fer.train(
        max_iter=config.training_cycles,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    # now we get the results or do whatever we want with them
    results_bos , _ , local_en_bos = system_bos.sample(config.nsamples, nchains=config.nchains, seed=config.seed)
    results_fer , _ , local_en_fer  = system_fer.sample(config.nsamples, nchains=config.nchains, seed=config.seed)


    block_size = 1000 # this is the block size for the blocking method

    energy_bos , variance_bos = system_bos.blocking_method(local_en_bos , block_size )
    energy_fer , variance_fer = system_fer.blocking_method(local_en_fer , block_size )


    energies_bosons.append(energy_bos)
    energies_fermions.append(energy_fer)
    variances_bosons.append(variance_bos)
    variances_fermions.append(variance_fer)
    



np.savetxt("data_analysis/bosons_energies.dat", np.array(energies_bosons))
np.savetxt("data_analysis/fermion_energies.dat", np.array(energies_fermions))
np.savetxt("data_analysis/n_particles.dat", n_particles)
np.savetxt("data_analysis/bosons_variances.dat", np.array(variances_bosons))
np.savetxt("data_analysis/fermion_variances.dat", np.array(variances_fermions))




