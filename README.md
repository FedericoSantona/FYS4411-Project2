# Template for students undertaking Computational Physics II 

This is a Python template for the v2024 edition of the course FYS4411. The course page can be found [here](https://www.uio.no/studier/emner/matnat/fys/FYS4411/v24/index.html).

### General information

Originally, the course was structured only in C++ (take a look at https://github.com/mortele/variational-monte-carlo-fys4411).
One of the main reasons for this, beyond learning modern C++, is the need for fast computations in Markov Chain Monte Carlo methods.

However, with JAX parallelizations and just-in-time compilation, it is possible to achieve comparable results in Python and therefore we are making this template available.
Whether you are using this template or just Python for the course projects, we **strongly** recommend JAX. Take a close look at the [documentation](https://jax.readthedocs.io/en/latest/index.html)

Parts of this template were inspired by https://github.com/nicolossus/FYS4411-Project2 (thank you!)

Using this template, you are supposed to fill some gaps in the code. Another possibility is even to start from scratch and use this as a more vague template.
Here you can find suggestions that can seem rigid in the code structure. Feel free to not follow those.

For example, the way we use just-in-time compilation and some function closures is just a suggestion to ensure the purity of some functions, but there might be other ways. 
Take a look at [this other material](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).
Another example where you might want to change things: maybe you do not want to have to pass the wavefunction object to the Hamiltonian class. Feel free to change those details if you don't like the global structure.


### Requirements
We suggest using [Poetry](https://python-poetry.org/) to manage project dependencies (as is clear from pyproject.toml and all the dependencies are there). Feel free to not use this as [pip supports installing .toml dependencies natively](https://stackoverflow.com/questions/62408719/download-dependencies-declared-in-pyproject-toml-using-pip).


### Getting initial results
If you are in a UNIX-based system, go to `/src/` from the terminal and type `pwd`.

Add that full src path at the beginning of the `vmc_playground.py` file, in 

```
sys.path.append("Full/Path/For/Src")
```

You can get initial results by going to `src/simulation_scripts/` and running 
```
python3 vmc_playground.py
```


*Disclaimer:* note those results do not make physical sense as you need to complete the functions and code bits yourself.
