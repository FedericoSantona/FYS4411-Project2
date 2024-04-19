# Config
output_filename = "../data/vmc_playground.csv"
import numpy as np

nparticles = 2
dim = 3
n_hidden = 10
init_scale = 1 / np.sqrt(nparticles * dim * n_hidden)
nsamples =  int(2**12) #  2**18 = 262144
scale = 1+ (dim-1)*0.1
nchains = 1# number of Markov chains. When you parallelize, you can set this to the number of cores. Note you will have to implement this yourself.
eta = 0.1
tol = 1e-6  #tolerance for the size of the gradient
training_cycles = 1000 # this is cycles for the ansatz
mcmc_alg = "m" # eiteer "mh" or "m"
backend = "numpy" # or "numpy" but jax should go faster because of the jit
optimizer = "adam"
hamiltonian = "ho" # either ho or eo 
interaction = "Coulomb" # either Coulomb or None
radius =  0 #0.0043
batch_size = 600
detailed = True
wf_type = "vmc" 
seed = 142



#only important for Metropolis hastings:

time_step = 0.05
diffusion_coeff = 0.5

