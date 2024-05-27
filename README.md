# Project 2 in FYS4411: Computational Physics II
The course page can be found [here](https://www.uio.no/studier/emner/matnat/fys/FYS4411/v24/index.html).

### General information
This repository contains all the code written in Python to build a VMC algorithm, following closely the project tasks presented [here](https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/Projects/2024/Project2/Project2ML/pdf/Project2ML.pdf). The general structure of our program has been inherited from [project 1](https://github.com/FedericoSantona/Repo), which was based to a great extent on the skeleton code provided by our remarkable TA, Daniel Haas. The skeleton code can be found [here](https://github.com/Daniel-Haas-B/FYS4411-Template?tab=readme-ov-file).


The implementation was done for both [JAX](https://jax.readthedocs.io/en/latest/index.html) and [NumPy](https://numpy.org), both discussed in project one. Further, the second project revolved around using the old machinery, which can be read about in project 1, with the Restricted Boltzmann Machine as a trial wave-function, later we also added a implementation of a more complex neural network as a possible trial wave-function. As in project 1, we've made the VMC algorithm to run with regular metropolis sampling, as well as using the Metro-Hastings (importance sampling). Furthermore, we've (also again) implemented deepest descent optimization to tune our variational parameters. 



### Requirements
Necessary dependencies can be found in the pyproject.toml file - and is easily handled by [Poetry](https://python-poetry.org/). Feel free to install them manually using Pip if you dislike simplicity (or have any other reason to not use Poetry).

### Running the VMC algorithm
Running our VMC implementation is simple, and straightforward. You can choose your system specific parameters inside the config.py file (located in src/simulation_scripts), 
which offers the following tune-able parameters to change the quantum system:
- Number of particles.
- Number of dimensions.
- Number of hidden nodes.
- Initialisaton scale of variational parameters.
- Harmonic oscillator frequency.
- A parameter which lets us choose the relation between the wave-function and Boltzmann machine.
- The particle type.
- The max degree.
- Number of samples.
- Sampling scale.
- Number of chains (for parallelisation). NOTE: Number of samples will then be the same _on each chain_
- MCMC algorithm of choice (with/without importance sampling).
- Using JAX or Numpy .
- Learning rate (eta) and number of training cycles.
- Termination tolerance for optimisation.
- Batch size in the training and number of training cycles
- Optimisation scheme
- Type of Hamiltonian (elliptic or harmonic)
- Interaction (None or Coulomb)

Also for the Importance sampling we have two additional parameters
- Time step.
- Diffusion coefficient.

After setting these parameters to reflect the quantum system one wishes to study, the entire program is ran using '>  ./vmc.sh' (located in FYS4411-Project2/). This will automatically start the VMC algorithm through src/simulation_scripts/vmc_playground.py. Different flags following ./vmc.sh will have the following effect
- No flag or vmc: Runs src/simulation_scripts/vmc_playground.py
- grid: Runs src/simulation_scripts/vmc_playground.py
- int: Runs src/simulation_scripts/interaction_vs_particles.py
- energy: Runs src/simulation_scripts/energy_particle.py
- boot: Runs src/simulation_scripts/bootstrap.py
- plot: Runs src/simulation_scripts/plot_producer.py

 If any other flag is put, denoted "$1", vmc.sh will attempt to run src/simulation_scripts/"$1" in the terminal. 
