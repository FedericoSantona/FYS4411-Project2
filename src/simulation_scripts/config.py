# Config
output_filename = "../data/vmc_playground.csv"

nparticles = 1
dim = 1
nsamples =   int(2**12) #  2**18 = 262144
scale = 1 + dim*0.1
nchains = 4 # number of Markov chains. When you parallelize, you can set this to the number of cores. Note you will have to implement this yourself.
eta = 0.1
training_cycles = 5000# this is cycles for the ansatz
mcmc_alg = "m"
backend = "numpy" # or "numpy" but jax should go faster because of the jit
optimizer = "gd"
batch_size = 100
detailed = True
wf_type = "vmc"
seed = 142
alpha = 0.5
beta = 1