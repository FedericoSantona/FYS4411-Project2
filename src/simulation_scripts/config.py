# Config
output_filename = "../data/vmc_playground.csv"

nparticles = 1
dim = 1
nsamples =  int(2**13) #  2**18 = 262144
scale = 1 + (dim-1)*0.1
nchains = 1# number of Markov chains. When you parallelize, you can set this to the number of cores. Note you will have to implement this yourself.
eta = 0.01
training_cycles = 5000# this is cycles for the ansatz
mcmc_alg = "m" # either "mh" or "m"
backend = "numpy" # or "numpy" but jax should go faster because of the jit
optimizer = "gd"
hamiltonian = "ho" # either ho or eo 
batch_size = 100
detailed = True
wf_type = "vmc"
seed = 142
alpha = 0.6
beta = 2.35


#only important for Metropolis hastings:

time_step = 0.05
diffusion_coeff = 0.5

