# Config
output_filename = "../data/vmc_playground.csv"
import numpy as np

nparticles = 2
dim = 2
n_hidden = 2
init_scale =  0.1


omega = 1 #Ho frequency

WF_scale = 1 # This is the decider between using psi =sqrt(F) [2] or psi = F [1] 

nsamples =  int(2**16) #  2**18 = 262144
scale = 1+ (dim-1)*0.1
nchains = 4# number of Markov chains
mcmc_alg = "m" # eiteer "mh" or "m"
backend = "numpy" # or "numpy" but jax should go faster because of the jit

eta = 1e-3
tol = 1e-8  #tolerance for the size of the gradient
training_cycles = 1000 # this is cycles for the ansatz
optimizer = "gd"
batch_size = 800


hamiltonian = "ho" # either ho or eo 
interaction = "None" # either Coulomb or None
radius =  0 #0.0043



detailed = True
wf_type = "vmc" 
seed = 142


#only important for Metropolis hastings:

time_step = 0.05
diffusion_coeff = 0.5

