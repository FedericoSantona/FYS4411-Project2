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
    def __init__(self, alg_inst, hamiltonian, rng, scale, n_particles, dim, seed, log, logger=None, logger_level="INFO", backend="numpy", time_step=0.01, diffusion_coeff=0.5):
        
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
        


        #print("initial_positions", initial_positions)

        # Use the current positions to generate the quantum force
        quantum_force_current = self.quantum_force(initial_positions)

        # Generate a proposal move
        next_gen = advance_PRNG_state(seed , state.delta)
        rng = self._rng(next_gen)

        # Generate a proposal move
        #eta = rng.normal(loc= 0 , scale = self.scale)
        eta = rng.normal(loc=0, scale=1, size=(self._N, self._dim))


        proposed_positions = initial_positions + self.diffusion_coeff * quantum_force_current * self.time_step + eta * (self.backend.sqrt(self.time_step))
        
        #print("proposed_positions", proposed_positions)
        
        # Calculate the quantum force for the proposed positions
        quantum_force_proposed = self.quantum_force(proposed_positions)

        # Calculate wave function squared for current and proposed positions
        prob_current = wf_squared(initial_positions)
        prob_proposed = wf_squared(proposed_positions)

        # Calculate the q value
        
        
        q_value = self.q_value(initial_positions, proposed_positions, quantum_force_current, quantum_force_proposed, self.diffusion_coeff, self.time_step, prob_current, prob_proposed)
        #breakpoint()
        #print("q_value", q_value.shape)
        
        # Decide on acceptance
        accept = rng.random(self._N) <  self.backend.exp(q_value)
        accept = accept.reshape(-1, 1)

       # print("accept", accept)

        # Update positions based on acceptance
        new_positions, new_logp, n_accepted = self.accept_func(n_accepted=n_accepted, accept=accept, initial_positions=initial_positions, proposed_positions=proposed_positions, log_psi_current=prob_current, log_psi_proposed=prob_proposed)
        
        #print("new_positions", new_positions)


        # Create new state
        new_state = State(positions=new_positions, logp=new_logp, n_accepted=n_accepted, delta=state.delta + 1)

        return new_state

    def quantum_force(self, positions):

        # the quantum force is 2 * the gradient of the log of the wave function
        #print("grad", self.alg_inst.grad_wf(positions))
        return 2* self.alg_inst.grad_wf(positions)
    

    def q_value(self, r_old, r_new, F_old, F_new, D, delta_t , wf2_old, wf2_new):


        beta = r_new - r_old
        
        squared_term = D*delta_t*0.25 *(self.backend.sum(F_old**2 , axis=1 ) - self.backend.sum(F_new**2 , axis = 1))

        #THE PROBLEM HERE IS HOW WE SUM THE BETA*F BECAUSE WE NEED TO REDUCE 
        #THE DIMENSIONALITY TO ( N PARTICLES, 1) BUT WE HAVE TO FIND A WAY THAT 
        # MAKES SENSE DO IT 

        
        linear_term = 0.5 *self.backend.sum(beta * (F_new + F_old), axis= 1)
        
        q_value =  squared_term - linear_term  + wf2_new - wf2_old

        return  q_value 
    

    def accept_func(self, n_accepted, accept, initial_positions, proposed_positions, log_psi_current, log_psi_proposed):
        # accept is a boolean array, so you can use it to index directly
        new_positions = np.where(accept, proposed_positions, initial_positions)
        new_logp = np.where(accept, log_psi_proposed, log_psi_current)

        # Count the number of accepted moves
        n_accepted += np.sum(accept)

        return new_positions, new_logp, n_accepted


"""
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
        
"""

    