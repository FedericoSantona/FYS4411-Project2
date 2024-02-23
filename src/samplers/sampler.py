import copy

import jax
import jax.numpy as jnp
import numpy as np
from qs.utils import check_and_set_nchains # we suggest you use this function to check and set the number of chains when you parallelize
from qs.utils import generate_seed_sequence
from qs.utils import State
from tqdm.auto import tqdm  # progress bar


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class Sampler:
    def __init__(self, rng, scale, logger=None , backend="numpy"):

        self._logger = logger
        self.results = None
        self._rng = rng
        self.scale = scale
        self._backend = backend

        match backend:
            case "numpy":
                self.backend = np
                self.la = np.linalg
                #self.accept = accept_numpy
            case  "jax":
                self.backend = jnp
                self.la = jnp.linalg
                #self.accept = accept_jax()
                # You might also be able to jit some functions here
            case _: # noqa
                raise ValueError("Invalid backend:", backend)


    def sample(self, wf, state, nsamples, nchains=1, seed=None):
        """ 
        Will call _sample() and return the results
        We set it this way because if want to be able to parallelize the sampling, each process will call _sample() and return the results to the main process.
        """
        
        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(seed, nchains)
        if nchains == 1:
            chain_id = 0

        else:
            # Parallelize
            pass


        self._sampling_performed_ = True
        if self._logger is not None:
            self._logger.info("Sampling done")

        return self._results

    def _sample(self, wf, nsamples, state, scale, seed, chain_id):
        """To be called by process. Here the actual sampling is performed."""

        if self._logger is not None:
            t_range = tqdm(
                range(nsamples),
                desc=f"[Sampling progress] Chain {chain_id+1}",
                position=chain_id,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(nsamples)

        # Config
        state = State(state.positions, state.logp, 0, state.delta)
        energies = np.zeros(nsamples)

        for i in t_range:
            # this is where you call the step method of the specific sampler (metropolis, metropolis-hastings, etc.)
            # then from the new state you calculate the local energies 
            pass 
        if self._logger is not None:
            t_range.clear()
                                
                                        
        # calculate energy, error, variance, acceptance rate, and other things you want to display in the results
    

        sample_results = {
            "chain_id": chain_id,
            "energy": None,
            "std_error": None,
            "variance": None,
            "accept_rate": None,
            "scale": None,
            "nsamples": nsamples,
        }

        return sample_results, energies

    def step(self, wf, state, seed):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def accept_jax(self, n_accepted , accept,  initial_positions , proposed_positions ,log_psi_current,  log_psi_proposed):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def accept_numpy(self, n_accepted , accept,  initial_positions , proposed_positions ,log_psi_current,  log_psi_proposed):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement this method.")
    

    def set_hamiltonian(self, hamiltonian):

        self.hamiltonian = hamiltonian
