# Config
output_filename = "../data/vmc_playground.csv"
import numpy as np

nparticles = 1
dim = 1
init_scale = 1 / np.sqrt(nparticles * dim )
n_hidden = 1
nsamples =  int(2**16) #  2**18 = 262144
scale = 1 + (dim-1)*0.1
nchains = 1# number of Markov chains. When you parallelize, you can set this to the number of cores. Note you will have to implement this yourself.
eta = 0.001
training_cycles = 1000 # this is cycles for the ansatz
mcmc_alg = "mh" # eiteer "mh" or "m"
backend = "numpy" # or "numpy" but jax should go faster because of the jit
optimizer = "gd"
hamiltonian = "ho" # either ho or eo 
interaction = "None" # either Coulomb or None
radius =  0#0.0043
batch_size = 200
detailed = True
wf_type = "vmc" 
seed = 142



#only important for Metropolis hastings:

time_step = 0.5
diffusion_coeff = 0.5

