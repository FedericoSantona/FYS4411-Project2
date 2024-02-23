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
    def __init__(self, alg_inst, hamiltonian, rng, scale, n_particles, dim, seed, log, logger=None, logger_level="INFO", backend="Numpy", time_step=0.01, diffusion_coeff=0.5):
        
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
            self.accept_fn = self.accept_numpy
            self.backend = np
        elif self._backend == "jax":
            self.accept_fn = self.accept_jax
            self.backend = jnp
        else:
            print("back end not supported !!!!!!")

        super().__init__(alg_inst, hamiltonian, log, rng, scale, logger, backend)
        


    def step(self, n_accepted, wf_squared, state, seed):
        """One step of the importance sampling Metropolis-Hastings algorithm."""


        initial_positions = state.positions
       
        # Use the current positions to generate the quantum force
        quantum_force_current = self.quantum_force(initial_positions)

        # Generate a proposal move
        next_gen = advance_PRNG_state(seed , state.delta)
        rng = self._rng(next_gen)

        # Generate a proposal move
        eta = rng.normal(loc= initial_positions , scale =  self.scale)

        proposed_positions = initial_positions + self.diffusion_coeff * quantum_force_current * self.time_step + eta * (self.backend.sqrt(self.time_step))

        # Calculate the quantum force for the proposed positions
        quantum_force_proposed = self.quantum_force(proposed_positions)

        # Calculate wave function squared for current and proposed positions
        prob_current = wf_squared(initial_positions)
        prob_proposed = wf_squared(proposed_positions)

        # Calculate the Greens function for acceptance probability
        G_forward , G_reverse = self.greens_function(initial_positions, proposed_positions, quantum_force_current, quantum_force_proposed, self.diffusion_coeff, self.time_step)

        # Calculate acceptance probability in log domain
        log_accept_prob = prob_proposed + G_reverse - prob_current - G_forward

        # Decide on acceptance
        accept = rng.random(initial_positions.shape[0]) < self.backend.exp(log_accept_prob)
        accept = accept.reshape(-1, 1)

        # Update positions based on acceptance
        new_positions, new_logp, n_accepted = self.accept_fn(n_accepted=n_accepted, accept=accept, initial_positions=initial_positions, proposed_positions=proposed_positions, log_psi_current=prob_current, log_psi_proposed=prob_proposed)

        # Create new state
        new_state = State(positions=new_positions, logp=new_logp, n_accepted=n_accepted, delta=state.delta + 1)

        return new_state

    def quantum_force(self, positions):

        # the quantum force is 2 * the gradient of the log of the wave function
        
        return 2* self.alg_inst.grad_wf(positions)

    def greens_function(self, r_old, r_new, F_old, F_new, D, delta_t):
        # Calculate the drift terms
        drift_old = D * F_old * delta_t
        drift_new = D * F_new * delta_t

        # Compute the squared distance terms for the forward and reverse moves
        forward_move = self.backend.linalg.norm(r_new - r_old - drift_old)**2
        reverse_move = self.backend.linalg.norm(r_old - r_new - drift_new)**2

        # Compute the Green's functions for the forward and reverse moves in the log domain
        G_forward = -forward_move / (4 * D * delta_t)
        G_reverse = -reverse_move / (4 * D * delta_t)

        return G_forward, G_reverse


    def accept_jax(self, n_accepted, accept, initial_positions, proposed_positions, log_psi_current, log_psi_proposed):
            # Use where to choose between the old and new positions/probabilities based on the accept array
            new_positions = jnp.where(accept, proposed_positions, initial_positions)
            new_logp = jnp.where(accept, log_psi_proposed, log_psi_current)

            # Count the number of accepted moves
            n_accepted += jnp.sum(accept)

            return new_positions, new_logp, n_accepted

    def accept_numpy(self, n_accepted, accept, initial_positions, proposed_positions, log_psi_current, log_psi_proposed):
        # accept is a boolean array, so you can use it to index directly
        new_positions = np.where(accept, proposed_positions, initial_positions)
        new_logp = np.where(accept, log_psi_proposed, log_psi_current)

        # Count the number of accepted moves
        n_accepted += np.sum(accept)

        return new_positions, new_logp, n_accepted
        
       