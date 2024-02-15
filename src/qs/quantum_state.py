# import copy
import sys
import warnings

sys.path.insert(0, "/mnt/c/Users/annar/OneDrive/Desktop/FYS4411/Project1_python/FYS4411-Template/src")

from qs.utils import errors
from qs.utils import generate_seed_sequence
from qs.utils import setup_logger
from qs.utils import State
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


import numpy as np
import pandas as pd

from qs.models import VMC

from numpy.random import default_rng
from tqdm.auto import tqdm

from physics.hamiltonians import HarmonicOscillator as HO

from samplers.metropolis import Metropolis as Metro

from  optimizers.gd import Gd as gd_opt

warnings.filterwarnings("ignore", message="divide by zero encountered")


class QS:
    def __init__(
        self,
        backend="numpy",
        log=True,
        logger_level="INFO",
        rng=None,
        seed=None,
    ):
        """Quantum State
        It is conceptually important to understand that this is the system.
        The system is composed of a wave function, a hamiltonian, a sampler and an optimizer.
        This is the high level class that ties all the other classes together.
        """

        self._check_logger(log, logger_level)
        self.backend = backend
        
        self._log = log
        self.hamiltonian = None
        self.mcmc_alg = None
        self._optimizer = None
        self.wf = None
        self.logp = None
        self._seed = seed
        self.logger_level = logger_level
        self.logger = setup_logger(self.__class__.__name__, level=logger_level) if self._log else None
        self._backend = backend
        
        if rng is None :
            # If no RNG is provided but a seed is, initialize a new RNG with the seed.
            self.rng = default_rng
        else:
            # If neither an RNG nor a seed is provided, initialize a new RNG without a specific seed.
            self.rng = rng
        # Suggestion of checking flags
            
        match backend:
            case "numpy":
                self.backend = np
                self.la = np.linalg
            case  "jax":
                self.backend = jnp
                self.la = jnp.linalg
                # You might also be able to jit some functions here
            case _: # noqa
                raise ValueError("Invalid backend:", backend)
        

        self._is_initialized_ = False
        self._is_trained_ = False
        self._sampling_performed = False

    def set_wf(self, wf_type, nparticles, dim, **kwargs):
        """
        Set the wave function to be used for sampling.
        For now we only support the VMC.
        Successfully setting the wave function will also initialize it 
        (this is because we expect the VMC class to initialize the variational parameters but you have to implement this yourself).
        """
    
        self._N = nparticles
        self._dim = dim
        self._wf_type = wf_type

        self.alg = VMC(self._N, self._dim, rng=self.rng, log=self._log, logger=self.logger , seed=self._seed , logger_level=self.logger_level, backend=self._backend )
        self.alg._initialize_vars(self._N, self._dim, self._log, self.logger, self.logger_level)

        self.alpha = self.alg.params.get("alpha")
        #activate the jit functions
        self.alg._jit_functions()
        

        if self._wf_type == "vmc":
           
            self.wf = self.alg.wf #might need positions
            self.logp = self.alg.prob # I initialize also this because why not
        else:
            raise ValueError("Invalid wave function type, should be 'vmc'")

        self._is_initialized_ = True


    def set_hamiltonian(self, type_, int_type, **kwargs):
        """
        Set the hamiltonian to be used for sampling.
        For now we only support the Harmonic Oscillator.

        Hamiltonian also needs to be propagated to the sampler if you at some point collect the local energy there.
        """
        self.int_type = int_type

        vmc_instance = self.alg


        if type_ == "ho":
            self.hamiltonian = HO(vmc_instance, self._N, self._dim,  self._log, self.logger, self._seed, self.logger_level, self.int_type, self._backend)
        else:
            raise ValueError("Invalid Hamiltonian type, should be 'ho'")
        # check HO script


    def set_sampler(self, mcmc_alg, scale=0.5):
        """
        Set the MCMC algorithm to be used for sampling.
        """
        self.mcmc_alg = mcmc_alg
        self._scale = scale
        
        vmc_instance = self.alg

        if self.mcmc_alg == "m":

            self.sampler = Metro(  vmc_instance, self.rng ,self._scale ,self._N , self._dim, self._seed, self._log,  self.logger , self.logger_level, self._backend) 

        # check metropolis sampler script



    def set_optimizer(self, optimizer, eta, **kwargs):
        """
        Set the optimizer algorithm to be used for param update.
        """
        self._eta = eta
    
        if optimizer == "gd":
           self._optimizer = gd_opt(eta=eta)
        else:
            raise ValueError("Invalid optimizer type, should be 'gd'")

    
    def train(self, max_iter,  batch_size, seed ,  **kwargs):
        """
        Train the wave function parameters.
        Here you should calculate sampler statistics and update the wave function parameters based on the derivative of the (statistical) local energy.
        """
        self._is_initialized()
        self._training_cycles = max_iter
        self._training_batch = batch_size

        # Define evaluation interval
        eval_interval = max_iter // 10  # Example: Evaluate every 10% of max_iter
    

        if self._log:
            t_range = tqdm(
                range(max_iter),
                desc="[Training progress]",
              #  position=0,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(max_iter)

        

        for iteration in t_range:

           
            # Sample data in batches
            _, sampled_positions, local_energies = self.sample(nsamples=batch_size, nchains=1, seed=seed)
            
            # Compute gradients for the batch
            grads_alpha = self.alg.grads(sampled_positions) 
            

            # Update alpha using the computed gradients and the optimizer
            self.alpha = self.backend.array(self._optimizer.step(self.alpha, grads_alpha))
    
            """
            # Optionally: Log intermediate results or perform evaluations
            if iteration % eval_interval == 0 or iteration == max_iter - 1:
                # Compute current average energy and its standard deviation
                avg_energy = jnp.mean(local_energies)
                std_dev_energy = jnp.std(local_energies)
                
                print("alpha" , self.alpha)
                print("grads_alpha" , grads_alpha)
                #print("position" , sampled_positions)
                # Logging
                if self._log:
                    tqdm.write(f"Iteration {iteration}: Avg Energy = {avg_energy:.4f}, StdDev = {std_dev_energy:.4f}  ")
                
            # Additionally, you can implement any model validation or checkpointing here
            """
            
                
            
            
        self._is_trained_ = True
        if self.logger is not None:
            self.logger.info("Training done")


    def sample(self, nsamples, nchains=1, seed=None):
        """helper for the sample method from the Sampler class"""

        
        self._is_initialized() # check if the system is initialized
        
       
        # Initialize state for the sampler
        #initial_positions = self.alg.state.positions # the position is initialized in the vmc class
        #initial_state = State(positions=initial_positions, logp = self.logp,  n_accepted=0, delta=0)
        #initial_state = self.alg.state

        #print("this is the initial state" , self.alg.state.positions, self.alg.state.logp, self.alg.state.n_accepted, self.alg.state.delta)

        sampled_positions = []
        local_energies = []  # List to store local energies
        total_accepted = 0  # Initialize total number of accepted moves
        
        

        for _ in range(nsamples):
            # Perform one step of the MCMC algorithm
            
            new_state  = self.sampler.step(total_accepted, self.wf, self.alg.state, self._seed )
            
            total_accepted = new_state.n_accepted

            self.alg.state = new_state

            # Calculate the local energy

            E_loc = self.hamiltonian.local_energy(self.wf, new_state.positions)
            
            local_energies.append(E_loc)  # Store local energy
            
            # total_accepted += accept  # Accumulate total accepted moves

            # Update the initial state with the new state for the next iteration
            #initial_state = new_state
           

            # Store sampled positions and calculate acceptance rate
            sampled_positions.append(new_state.positions)
        
        # Calculate acceptance rate
        acceptance_rate = total_accepted / nsamples

        # Compute statistics of local energies
        mean_energy = np.mean(local_energies)
        std_error = np.std(local_energies) / np.sqrt(nsamples)
        variance = np.var(local_energies)


        # Suggestion of things to display in the results
        system_info = {
            "nparticles": self._N,
            "dim": self._dim,
            "eta": self._eta,
            "mcmc_alg": self.mcmc_alg,
            "training_cycles": self._training_cycles,
            "training_batch": self._training_batch,
            "Opti": self._optimizer.__class__.__name__,
        }

        system_info = pd.DataFrame(system_info, index=[0])

        #OBS: this should actually be returned from the sampler sample method. This is as is below just a placeholder
        # Update the sample_results dictionary
        sample_results = {
            "chain_id": None,
            "energy": mean_energy,
            "std_error": std_error,
            "variance": variance,
            "accept_rate": acceptance_rate,
            "scale": None,
            "nsamples": nsamples,
        }
        sample_results = pd.DataFrame(sample_results, index=[0])


        system_info_repeated = system_info.loc[
            system_info.index.repeat(len(sample_results))
        ].reset_index(drop=True)

        self._results = pd.concat([system_info_repeated, sample_results], axis=1)

        #print("this is the sampled position" , sampled_positions)

        local_energies = np.array(local_energies)
        sampled_positions = np.array(sampled_positions)

        return self._results, sampled_positions, local_energies
    

    def _is_initialized(self):
        if not self._is_initialized_:
            msg = "A call to 'init' must be made before training"
            raise errors.NotInitialized(msg)

    def _is_trained(self):
        if not self._is_trained_:
            msg = "A call to 'train' must be made before sampling"
            raise errors.NotTrained(msg)

    def _sampling_performed(self):
        if not self._is_trained_:
            msg = "A call to 'sample' must be made in order to access results"
            raise errors.SamplingNotPerformed(msg)

    def _check_logger(self, log, logger_level):
        if not isinstance(log, bool):
            raise TypeError("'log' must be True or False")

        if not isinstance(logger_level, str):
            raise TypeError("'logger_level' must be passed as str")