# Config
output_filename = "../data/vmc_playground.csv"
import numpy as np

nparticles = 5
dim = 3
n_hidden = 10
init_scale = 1 / np.sqrt(nparticles * dim * n_hidden) * 0.01

nsamples =  int(2**12) #  2**18 = 262144
scale = 1+ (dim-1)*0.1
nchains = 4# number of Markov chains
mcmc_alg = "mh" # eiteer "mh" or "m"
backend = "jax" # or "numpy" but jax should go faster because of the jit

eta = 1e-1
tol = 1e-8  #tolerance for the size of the gradient
training_cycles = 1000 # this is cycles for the ansatz
optimizer = "adam"
batch_size = 400


hamiltonian = "ho" # either ho or eo 
interaction = "None" # either Coulomb or None
radius =  0 #0.0043



detailed = True
wf_type = "vmc" 
seed = 142


#only important for Metropolis hastings:

time_step = 0.05
diffusion_coeff = 0.5

