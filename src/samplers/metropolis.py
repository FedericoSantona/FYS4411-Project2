import jax.numpy as jnp
import numpy as np
from jax import jit, ops
from jax import lax
import jax
import jax.random as random
from qs.utils import State
from qs.utils import advance_PRNG_state
from .sampler import Sampler
from qs.models.vmc import VMC
from qs.utils.parameter import Parameter


class Metropolis(Sampler):
    def __init__(
        self,
        alg_inst,
        hamiltonian,
        rng,
        scale,
        n_particles,
        dim,
        seed,
        log,
        logger=None,
        logger_level="INFO",
        backend="Numpy",
    ):
        # Initialize the VMC instance
        # Initialize Metropolis-specific variables
        self.step_method = self.step
        self._seed = seed
        self._N = n_particles
        self._dim = dim
        
        super().__init__(alg_inst, hamiltonian, log, rng, scale, logger, backend)

    

    def step(self, wf_squared, state, seed):
        """One step of the random walk Metropolis algorithm."""
        initial_positions = state.positions
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)
        # Generate a proposal move

        proposed_positions_tot = rng.normal(loc=initial_positions , scale=self.scale)

        initial_positions = np.array(initial_positions)
        proposed_positions_tot = np.array(proposed_positions_tot)

        
        for i in range(self._N):
            
            proposed_positions = initial_positions.copy()
            


            #proposed_positions = jnp.where(i == jnp.arange(proposed_positions.shape[0])[:, None], proposed_positions_tot[i, :], proposed_positions)
            
            proposed_positions[i , :] = proposed_positions_tot[i , :]

            prop = proposed_positions
            # Calculate log probability densities for current and proposed positions
            prob_current = wf_squared(initial_positions)
            prob_proposed = wf_squared(proposed_positions)
            # Calculate acceptance probability in log domain
            log_accept_prob = prob_proposed - prob_current

            # Decide on acceptance
            accept = np.full(self._N, False, dtype=bool)
            accept[i] = rng.random() < np.exp(
                log_accept_prob
            )

            
            #breakpoint()
            new_positions, new_logp, n_accepted = self.accept_func(
                n_accepted=state.n_accepted,
                accept=accept,
                initial_positions=initial_positions,
                proposed_positions=proposed_positions,
                log_psi_current=prob_current,
                log_psi_proposed=prob_proposed,
            )
            init = initial_positions
            
            if accept[i]:
                initial_positions = new_positions

            #breakpoint()

            
        new_positions_tot = initial_positions
        # Create new state by updating state variables.
        state.logp = new_logp
        state.delta += 1
        state.n_accepted = n_accepted
        state.positions = new_positions_tot
        state.r_dist = new_positions[None, ...] - new_positions[:, None, :]

    def accept_func(
        self,
        n_accepted,
        accept,
        initial_positions,
        proposed_positions,
        log_psi_current,
        log_psi_proposed,
    ):
        
        accept = accept.reshape(-1, 1)
        # accept is a boolean array, so you can use it to index directly
        new_positions = np.where(accept, proposed_positions, initial_positions)
        new_logp = np.where(accept, log_psi_proposed, log_psi_current)

        # Count the number of accepted moves
        n_accepted += np.sum(accept)

        return new_positions, new_logp, n_accepted
