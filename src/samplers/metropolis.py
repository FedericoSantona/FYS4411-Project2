import jax.numpy as jnp
import jax.random as random
from qs.utils import State
from qs.utils import advance_PRNG_state
from .sampler import Sampler
from qs.models.vmc import VMC
from qs.utils.parameter import Parameter

class Metropolis(Sampler):
    def __init__(self, alg_inst, rng, scale, n_particles, dim, seed, log, logger=None, logger_level="INFO", backend="Numpy"):
        # Initialize the VMC instance
        
        # Initialize Metropolis-specific variables
        self.scale = scale
        self.logger = logger
        self._seed = seed
        self.rng = rng
        self._N = n_particles
        self._dim = dim
        self._log = log
        self.alg_inst = alg_inst
        
        
        super().__init__(rng, scale, logger)

    def step(self, wf, state, seed):

        """One step of the random walk Metropolis algorithm."""

        initial_positions = state.positions
        #initial_logp = state.logp

        #print("state BEFORE", initial_positions, initial_logp, state.n_accepted, state.delta)

        jax_key = advance_PRNG_state(seed , state.delta)

        proposal_key, accept_key = random.split(jax_key)

        # Generate a proposal move
        proposal_move = random.normal(proposal_key, initial_positions.shape) * self.scale

        # Calculate proposed positions
        proposed_positions = initial_positions + proposal_move

        print("proposal_move", proposal_move)

        print("proposed_positions", proposed_positions)

        print( "initial_positions.shape", initial_positions.shape)

        print("proposed_positions.shape", proposed_positions.shape)

        # Calculate log probability densities for current and proposed positions
        log_psi_current = wf(initial_positions)
        log_psi_proposed = wf(proposed_positions)

        print("log_psi_current", log_psi_current)
        print("log_psi_proposed", log_psi_proposed)

        # Calculate acceptance probability in log domain
        log_accept_prob = (log_psi_proposed - log_psi_current)

        print("log_accept_prob", log_accept_prob)
        print("log_accept_prob.shape", log_accept_prob.shape)

        # Decide on acceptance
        accept = random.uniform(accept_key, log_accept_prob.shape) < jnp.exp(log_accept_prob)


        print("accept", accept.shape, accept)

        # Update state based on acceptance
        new_positions = jnp.where(accept[:, None], proposed_positions, initial_positions)
        new_logp = jnp.where(accept, log_psi_proposed, log_psi_current)
        n_accepted = jnp.sum(accept)

        print("initial_positions", initial_positions)   
        print("new_positions", new_positions)

        # Create new state
        new_state = State(positions=new_positions, logp=new_logp, n_accepted=n_accepted, delta=state.delta + 1)

        #print("state AFTER", new_state.positions, new_state.logp, new_state.n_accepted, new_state.delta)

        return new_state