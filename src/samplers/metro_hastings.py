import jax.numpy as jnp
import numpy as np
from jax import jit, ops
from jax import lax
import jax.random as random
from qs.utils import State
from qs.utils import advance_PRNG_state
from .sampler import Sampler
from qs.models.vmc import VMC
from qs.utils.parameter import Parameter


class MetropolisHastings(Sampler):
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
        backend="numpy",
        time_step=0.01,
        diffusion_coeff=0.5,
    ):

        self.time_step = time_step
        self.diffusion_coeff = diffusion_coeff
        self._seed = seed
        self._N = n_particles
        self._dim = dim
        self._backend = backend
        self.alg_inst = alg_inst
        self.scale = scale
        self._rng = rng

        # The step method is bound to the instance
        self.step_method = self.step

        if self._backend == "numpy":
            self.backend = np
        elif self._backend == "jax":
            self.backend = jnp
            self.jit_functions()
        else:
            print("back end not supported !!!!!!")

        super().__init__(alg_inst, hamiltonian, log, rng, scale, logger, backend)

    def jit_functions(self):
        """Put the JAX jit wrapper on jittable functions inside the sampler"""
        self.importance_sampling_interior = jit(self.importance_sampling_interior)
    
    
    def step(self, wf_squared, state, seed):
        """One step of the importance sampling Metropolis-Hastings algorithm."""
        initial_positions = state.positions
        quantum_force_init = 2 * self.alg_inst.grad_wf(initial_positions)
        # Use the current positions to generate the quantum force
        # Generate a proposal move
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)
        # Generate a proposal move
        eta = rng.normal(loc=0, scale=1, size=(self._N, self._dim))
        proposed_positions = (
            initial_positions
            + self.diffusion_coeff * quantum_force_init * self.time_step
            + eta * (self.backend.sqrt(self.time_step))
        )
        # Calculate wave function squared for current and proposed positions
        prob_current = wf_squared(initial_positions)
        prob_proposed = wf_squared(proposed_positions)

        # Calculate the q - value
        q_value, proposed_positions = self.importance_sampling_interior(initial_positions,
                                                                        proposed_positions,
                                                                        quantum_force_init,
                                                                        prob_current,
                                                                        prob_proposed,
                                                                        self.diffusion_coeff,
                                                                        self.time_step)
        # Decide on acceptance
        accept = rng.random(self._N) < self.backend.exp(q_value)
        accept = accept.reshape(-1, 1)
        # Update positions based on acceptance
        new_positions, new_logp, n_accepted = self.accept_func(
            n_accepted=state.n_accepted,
            accept=accept,
            initial_positions=initial_positions,
            proposed_positions=proposed_positions,
            log_psi_current=prob_current,
            log_psi_proposed=prob_proposed,
        )
        # Create new state by updating state variables.
        state.logp = new_logp
        state.n_accepted = n_accepted
        state.delta += 1
        state.positions = new_positions


    def importance_sampling_interior(self,
                                     initial_positions,
                                     proposed_positions,
                                     q_force_init,
                                     prob_init,
                                     prob_proposed,
                                     D,
                                     dt):
        
        q_force_proposed = 2 * self.alg_inst.grad_wf(proposed_positions)
        
        # Calculate wave function squared for current and proposed positions
        
        v_init = proposed_positions - initial_positions - D * dt * q_force_init
        v_proposed = initial_positions - proposed_positions - D * dt * q_force_proposed

        Gfunc_init = -self.backend.sum(v_init **2, axis=1) / (D * dt * 4)
        Gfunc_proposed = -self.backend.sum(v_proposed **2, axis=1) / (D * dt * 4)

        q_value = Gfunc_proposed - Gfunc_init + prob_proposed - prob_init

        return q_value, proposed_positions
    
    def accept_func(
        self,
        n_accepted,
        accept,
        initial_positions,
        proposed_positions,
        log_psi_current,
        log_psi_proposed,
    ):
        # accept is a boolean array, so you can use it to index directly
        new_positions = np.where(accept, proposed_positions, initial_positions)
        new_logp = np.where(accept, log_psi_proposed, log_psi_current)

        # Count the number of accepted moves
        n_accepted += np.sum(accept)

        return new_positions, new_logp, n_accepted

