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

class Metropolis(Sampler):
    def __init__(self, alg_inst, hamiltonian , rng , scale, n_particles, dim, seed, log, logger=None, logger_level="INFO", backend="Numpy"):
        # Initialize the VMC instance
        
        # Initialize Metropolis-specific variables
        self._seed = seed
        self._N = n_particles
        self._dim = dim
        self.step_method = self.step

        super().__init__(alg_inst , hamiltonian , log,  rng, scale, logger, backend)

        if self._backend == "numpy":
            self.accept_fn = self.accept_numpy
        elif self._backend == "jax":
            self.accept_fn = self.accept_jax
        else:
            print("back end not supported !!!!!!")
        



    def step(self,  n_accepted , wf_squared, state, seed):

        """One step of the random walk Metropolis algorithm."""

        initial_positions = state.positions
       
        #print("initial_positions", initial_positions)
        next_gen = advance_PRNG_state(seed , state.delta)

       

        rng = self._rng(next_gen)


        # Generate a proposal move
        proposed_positions = rng.normal(loc= initial_positions , scale =  self.scale)

        
        
        # Calculate log probability densities for current and proposed positions
        prob_current = wf_squared(initial_positions)
        prob_proposed = wf_squared(proposed_positions)

        #print("log_psi_current", log_psi_current)
        #print("log_psi_proposed", log_psi_proposed)

        # Calculate acceptance probability in log domain
        log_accept_prob = (prob_proposed - prob_current)

        
        #print("log_accept_prob", log_accept_prob)
        #print("log_accept_prob.shape", log_accept_prob.shape)

        # Decide on acceptance
        
        accept = rng.random(initial_positions.shape[0]) < self.backend.exp(log_accept_prob)
        accept = accept.reshape(-1,1)

       

    
        new_positions ,new_logp , n_accepted = self.accept_fn(n_accepted= n_accepted , accept = accept,  initial_positions = initial_positions , proposed_positions = proposed_positions ,log_psi_current = prob_current,  log_psi_proposed = prob_proposed)
        

        #print("initial_positions", initial_positions)   
        #print("new_positions", new_positions)

        

        # Create new state
        new_state = State(positions=new_positions, logp=new_logp, n_accepted=n_accepted, delta=state.delta + 1)

        #print("state AFTER", new_state.positions, new_state.logp, new_state.n_accepted, new_state.delta)

        return new_state 
    
    
    #These should work better

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