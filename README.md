# Project 1 in FYS4411: Computational Physics II
The course page can be found [here](https://www.uio.no/studier/emner/matnat/fys/FYS4411/v24/index.html).

### General information
This repository contains all the code written in Python to build a VMC algorithm, following closely the project tasks presented [here](https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/Projects/2024/Project1/pdf/Project1.pdf). The general structure of our program has been inspired to a great extent by our remarkable TA, Daniel Haas, which provided an excellent skeleton for us to build our program using Python. This beauty can be found [here](https://github.com/Daniel-Haas-B/FYS4411-Template?tab=readme-ov-file).

We've made the VMC algorithm to run with regular metropolis sampling, as well as using the Metro-Hastings (importance sampling). Furthermore, we've implemented deepest descent optimization to tune our variational parameters. Most of our code is written in [JAX](https://jax.readthedocs.io/en/latest/index.html) - which offers a high performing compilation of our Python-code - as well as automatic differentiation. We have however, built most of the same code using [NumPy](https://numpy.org) aswell, mostly to benchmark, and for simplicity when creating our programs. 

### Requirements
Necessary dependencies can be found in the pyproject.toml file - and is easily handled by [Poetry](https://python-poetry.org/). Feel free to install them manually using Pip if you dislike simplicity (or have any other reason to not use Poetry).

### Running the VMC algorithm
Running our VMC implementation is simple, and straightforward. You can choose your system specific parameters inside the config.py file (located in src/simulation_scripts), 
which offers the following tune-able parameters to change the quantum system:
- Number of particles.
- Number of dimensions.
- Number of samples.
- Number of chains (for parallelisation). NOTE: Number of samples will then be the same _on each chain_
- Learning rate (eta) and number of training cycles
- Batch size in the training
- MCMC algorithm of choice (with/without importance sampling)
- Type of Hamiltonian (elliptic or harmonic)
- Interaction (None or Coulomb)
- Initial guess for the variational parameter, alpha

After setting these parameters to reflect the quantum system one wishes to study, the entire program is ran using '>  python vmc_playground.py' (file is also located in src/simulation_scripts). This will automatically start the VMC algorithm, and also nicely present some values we deem interesting (i.e the ground state energy, variance, std dev etc.)

