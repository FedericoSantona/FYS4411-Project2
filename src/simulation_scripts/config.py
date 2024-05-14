# Config
output_filename = "../data/vmc_playground.csv"
import numpy as np

nparticles = 2
dim = 3
n_hidden = 100
init_scale =  0.1


omega = 1 #Ho frequency

WF_scale = 1 # This is the decider between using psi =sqrt(F) [2] or psi = F [1] 

particle_type = "bosons" # either "bosons" or "fermions"
max_degree = nparticles // 2

nsamples =  int(2**16) #  2**18 = 262144
scale = 1+ (dim-1)*0.1
nchains = 4# number of Markov chains
mcmc_alg = "mh" # eiteer "mh" or "m"
backend = "jax" # or "numpy" but jax should go faster because of the jit

eta = 0.001
tol = 1e-8  #tolerance for the size of the gradient
training_cycles = 1000 # this is cycles for the ansatz
optimizer = "gd"
batch_size = 600


hamiltonian = "ho" # either ho or eo 
interaction = "None" # either Coulomb or None

detailed = True
wf_type = "vmc" 
seed = 142


#only important for Metropolis hastings:

time_step = 1
diffusion_coeff = 0.5
radius = 0

