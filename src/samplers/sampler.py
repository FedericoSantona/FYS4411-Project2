import copy

import jax
import jax.numpy as jnp
import numpy as np
from qs.utils import (
    check_and_set_nchains,
)  # we suggest you use this function to check and set the number of chains when you parallelize
from qs.utils import generate_seed_sequence
from qs.utils import State
from tqdm.auto import tqdm  # progress bar


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

        match backend:
            case "numpy":
                self.backend = np
                self.la = np.linalg
                # self.accept = accept_numpy
            case "jax":
                self.backend = jnp
                self.la = jnp.linalg
                # self.accept = accept_jax()
                # You might also be able to jit some functions here
            case _:  # noqa
                raise ValueError("Invalid backend:", backend)

    def sample(self, nsamples, nchains=1, seed=None):
        """
        Will call _sample() and return the results
        We set it this way because if want to be able to parallelize the sampling, each process will call _sample() and return the results to the main process.
        """

        nchains = check_and_set_nchains(nchains, self._logger)
        seeds = generate_seed_sequence(
            seed, nchains
        )  # YOU have to understand how to use it because you already do it in the MEtro step
        if nchains == 1:
            chain_id = 0

            self._results, self._sampled_positions, self._local_energies = self._sample(
                nsamples, chain_id
            )

        else:
            # Parallelize
            print("Value Error parallelization is not implemented!!!!!!")
            pass

        self._sampling_performed_ = True
        if self._logger is not None:
            self._logger.info("Sampling done")

        return self._results, self._sampled_positions, self._local_energies

    def _sample(self, nsamples, chain_id):
        """To be called by process. Here the actual sampling is performed."""
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

        sampled_positions = []
        local_energies = []  # List to store local energies
        total_accepted = 0  # Initialize total number of accepted moves

        for _ in t_range:  # Here use range(nsamples) if you train
            # Perform one step of the MCMC algorithm
            # Find the next state
            new_state = self.step(
                total_accepted, self.alg.prob, self.alg.state, self._seed
            )
            total_accepted = new_state.n_accepted
            self.alg.state = new_state

            # Calculate the local energy
            E_loc = self.hami.local_energy(self.alg.wf, new_state.positions)
            local_energies.append(E_loc)  # Store local energy
            # Store sampled positions and calculate acceptance rate
            sampled_positions.append(new_state.positions)

        if self._logger is not None:
            t_range.clear()

        # Calculate acceptance rate and convert lists to arrays
        acceptance_rate = total_accepted / (nsamples * self.alg._N)
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

    def accept_jax(
        self,
        n_accepted,
        accept,
        initial_positions,
        proposed_positions,
        log_psi_current,
        log_psi_proposed,
    ):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def accept_numpy(
        self,
        n_accepted,
        accept,
        initial_positions,
        proposed_positions,
        log_psi_current,
        log_psi_proposed,
    ):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def set_hamiltonian(self, hamiltonian):
        """Set the Hamiltonian for the sampler
        """
        self.hamiltonian = hamiltonian
