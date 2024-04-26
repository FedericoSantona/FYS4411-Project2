import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from qs.utils import (
    check_and_set_nchains,
)
from qs.utils import generate_seed_sequence
from qs.utils import sampler_utils
from qs.utils import State
from tqdm.auto import tqdm  # progress bar
import config



jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class Sampler:
    def __init__(self, alg, hamiltonian, log, rng, scale, logger=None, backend="numpy"):

        self._logger = logger
        self._log = log
        self.results = None
        self._rng = rng
        self.scale = scale
        self._backend = backend
        self.alg = alg
        self.hami = hamiltonian
        self.step_method = None
        self.n_training_cycles = config.training_cycles
        match backend:
            case "numpy":
                self.backend = np
                self.la = np.linalg
            case "jax":
                self.backend = jnp
                self.la = jnp.linalg
            case _:
                raise ValueError("Invalid backend:", backend)

    def sample(self, nsamples, nchains=1, seed=None):
        """
        Will call _sample() and return the results
        We set it this way because if want to be able to parallelize the sampling, each process will call _sample() and return the results to the main process.
        """

        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(
            seed, nchains
        )  # Generate seeds for each chain
        if nchains == 1:
            chain_id = 0

            self._results, self._sampled_positions, self._local_energies = self._sample(
                nsamples, chain_id
            )

        else:
            multi_sampler = sampler_utils.multiproc
            results, self._sampled_positions, self._local_energies = multi_sampler(
                self._sample, nsamples, nchains, seeds
            )
            self._results = pd.DataFrame(results)

        self._sampling_performed_ = True
        if self._logger is not None and self._log:
            self._logger.info("Sampling done")

        return self._results, self._sampled_positions, self._local_energies

    def _sample(self, nsamples, chain_id, seed=None):
        """To be called by process. Here the actual sampling is performed.

        Args:
        seed : int,
            Seed for the random number generator. The default is self._seed (what was initialized in the system class).
            We need to able to set the seed for each chain in the sampling process, otherwise we will get the same results for each chain.
        """
        if self._log:
            t_range = tqdm(
                range(nsamples),
                desc=f"[Sampling progress] Chain {chain_id+1}",
                position=chain_id,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(nsamples)
        # Set the seed for the chain
        if seed is None:
            seed = self._seed

        self.alg.state.n_accepted = 0
        sampled_positions = []
        local_energies = []  # List to store local energies
        for _ in t_range:  # Here use range(nsamples) if you train
            # Perform one step of the MCMC algorithm by updating the state parameters
            # WE DO NOT create a new state instance, as this is not necessary.
            self.step(self.alg.prob, self.alg.state, seed)
            E_loc = self.hami.local_energy( self.alg.state.positions)
            local_energies.append(E_loc)  # Store local energy
            sampled_positions.append(self.alg.state.positions)

        if self._logger is not None:
            pass
        """
        # Calculate acceptance rate and convert lists to arrays
        if config.training_cycles != 0 and nsamples != 0:
            acceptance_rate = self.alg.state.n_accepted / (
                nsamples * self.alg._N * self.n_training_cycles
            )
        else:
        """
        acceptance_rate = self.alg.state.n_accepted / (nsamples * self.alg._N)
        local_energies = self.backend.array(local_energies)
        sampled_positions = self.backend.array(sampled_positions)
        mean_positions = self.backend.mean(self.backend.abs(sampled_positions), axis=0)
        # Compute statistics of local energies
        mean_energy = self.backend.mean(local_energies)
        std_error = self.backend.std(local_energies) / self.backend.sqrt(nsamples)
        variance = self.backend.var(local_energies)
        # calculate energy, error, variance, acceptance rate, and other things you want to display in the results

        # Suggestion of things to display in the results
        sample_results = {
            "chain_id": chain_id,
            "energy": mean_energy,
            "std_error": std_error,
            "variance": variance,
            "accept_rate": acceptance_rate,
            "scale": self.scale,
            "nsamples": nsamples,
        }

        return sample_results, sampled_positions, local_energies

    def set_hamiltonian(self, hamiltonian):
        """Set the Hamiltonian for the sampler"""
        self.hamiltonian = hamiltonian
