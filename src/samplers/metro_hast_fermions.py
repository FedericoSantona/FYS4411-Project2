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


class MetropolisHastings_fermions(Sampler):
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

        #print("initial_positions", initial_positions)
        quantum_force_init = (2 * self.alg_inst.grad_wf(initial_positions)).reshape(self._N, self._dim)
        # Use the current positions to generate the quantum force
        # Generate a proposal move
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)
        # Generate a proposal move
        eta = rng.normal(loc=0, scale=1, size=(self._N , self._dim))
        proposed_positions_tot = (
            initial_positions
            + self.diffusion_coeff * quantum_force_init * self.time_step
            + eta * (self.backend.sqrt(self.time_step))
        )

        initial_positions = np.array(initial_positions)
        proposed_positions_tot = np.array(proposed_positions_tot)
        n_accepted = state.n_accepted

        for i in range(self._N):
            
            proposed_positions = initial_positions.copy()

            #proposed_positions = jnp.where(i == jnp.arange(proposed_positions.shape[0])[:, None], proposed_positions_tot[i, :], proposed_positions)
            
            proposed_positions[i , :] = proposed_positions_tot[i , :]

            # print("proposed_positions", proposed_positions)
            # Calculate wave function squared for current and proposed positions
            prob_current = wf_squared(initial_positions)
            prob_proposed = wf_squared(proposed_positions)

            # Calculate the q - value
            q_value, proposed_positions = self.importance_sampling_interior(
                initial_positions,
                proposed_positions,
                quantum_force_init,
                prob_current,
                prob_proposed,
                self.diffusion_coeff,
                self.time_step,
            )

            # Decide on acceptance
            accept = np.full(self._N, False, dtype=bool)
            accept[i] = rng.random() < (q_value)
            accept = accept.reshape(-1, 1)

            #print("accept", accept)
            # Update positions based on acceptance
            new_positions, new_logp = self.accept_func(
                accept=accept,
                initial_positions=initial_positions,
                proposed_positions=proposed_positions,
                log_psi_current=prob_current,
                log_psi_proposed=prob_proposed,
            )
            #ini = initial_positions
            if accept[i]:
                initial_positions = new_positions
                n_accepted += 1
                #print("accettata")
                #print(n_accepted)

            



        # Create new state by updating state variables.
        new_positions_tot = initial_positions
        state.logp = new_logp
        state.n_accepted = n_accepted
        state.delta += 1
        state.positions = new_positions_tot
        state.r_dist = new_positions[None, ...] - new_positions[:, None, :]

    def importance_sampling_interior(
        self,
        initial_positions,
        proposed_positions,
        q_force_init,
        prob_init,
        prob_proposed,
        D,
        dt,
    ):

        q_force_proposed = (2 * self.alg_inst.grad_wf(proposed_positions)).reshape(self._N, self._dim)

        

        # Calculate wave function squared for current and proposed positions
        v_init = proposed_positions - initial_positions - D * dt * q_force_init
        v_proposed = initial_positions - proposed_positions - D * dt * q_force_proposed


       #Now here should I sum along dimensions or sum it all? I think sum it all
        Gfunc_init = self.backend.exp(-self.backend.sum(v_init**2) / (D * dt * 4))
        Gfunc_proposed = self.backend.exp(-self.backend.sum(v_proposed**2) / (D * dt * 4))

       
        q_value = ( Gfunc_proposed  * prob_proposed) / (prob_init * Gfunc_init )

    
        return q_value, proposed_positions

    def accept_func(
        self,
        accept,
        initial_positions,
        proposed_positions,
        log_psi_current,
        log_psi_proposed,
    ):
        # accept is a boolean array, so you can use it to index directly
        new_positions = np.where(accept, proposed_positions, initial_positions)
        new_logp = np.where(accept, log_psi_proposed, log_psi_current)

        

        return new_positions, new_logp
