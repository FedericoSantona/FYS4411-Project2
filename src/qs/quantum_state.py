# import copy
import sys
import warnings

from qs.utils import errors
from qs.utils import generate_seed_sequence
from qs.utils import setup_logger
from qs.utils import State
from  qs.utils import advance_PRNG_state
from qs.utils import check_and_set_nchains
from samplers.sampler import Sampler
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
from physics.hamiltonians import EllipticOscillator as EO
from samplers.metropolis import Metropolis as Metro
from samplers.metro_hastings import MetropolisHastings as MetroHastings

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
        alpha = None,
        beta = None,
        time_step = None,
        diffusion_coeff = None
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
        self._init_alpha = alpha
        self.beta = beta
        self.time_step = time_step
        self.diffusion_coeff = diffusion_coeff
        
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
            

        print( "This calculation is done with the following backend: ", self._backend)
        

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

        self.alg = VMC(self._N, self._dim, rng=self.rng, log=self._log, logger=self.logger , seed=self._seed , logger_level=self.logger_level, backend=self._backend, alpha = self._init_alpha)
        self.alg._initialize_vars(self._N, self._dim, self._log, self.logger, self.logger_level)

        self.alpha = self.alg.params.get("alpha")
        #activate the jit functions
        # Should not be necessary, since this is called in the INIT phase of the VMC class.
        # self.alg._jit_functions()
        

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
        elif type_ == "eo":
            self.hamiltonian = EO(vmc_instance, self._N, self._dim,  self._log, self.logger, self._seed, self.logger_level, self.int_type, self._backend , self.beta)
        else:
            raise ValueError("Invalid Hamiltonian type, should be 'ho' o 'eo'")
        # check HO script


    def set_sampler(self, mcmc_alg, scale=0.5):
        """
        Set the MCMC algorithm to be used for sampling.
        """
        self.mcmc_alg = mcmc_alg
        self._scale = scale
        
        vmc_instance = self.alg
        hami = self.hamiltonian

        if self.mcmc_alg == "m":
            print("The chosen MCMC algorithm is the Metropolis algorithm")
            self.sampler = Metro(  vmc_instance, hami ,  self.rng ,self._scale ,self._N , self._dim, self._seed, self._log,  self.logger , self.logger_level, self._backend) 
        
        elif self.mcmc_alg == "mh":
            print("The chosen MCMC algorithm is the Metropolis-Hastings algorithm")
            self.sampler = MetroHastings(  vmc_instance, hami ,  self.rng ,self._scale ,self._N , self._dim, self._seed, self._log,  self.logger , self.logger_level, self._backend, self.time_step, self.diffusion_coeff)
        else:
            raise ValueError("Invalid MCMC algorithm type, should be 'm' or 'mh'")
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

    # This should be jittable, but this will be looked at when we start working on training.
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
            grads_alpha = self.backend.mean(self.alg.grads_E_loc(sampled_positions)) 

            grads_alpha = grads_alpha.reshape(-1 , 1)

            # Update alpha using the computed gradients and the optimizer
            #self.alpha = self.backend.array(self._optimizer.step(self.alpha, grads_alpha))
            # Ensure alpha and its gradient are iterables
            self.alpha = self.backend.array(self._optimizer.step([self.alpha], [grads_alpha]))[0]


            print("Alpha druing training", self.alpha)
           
            # Update the alpha in the Parameter instance
            self.alg.params.set("alpha", self.alpha)

              
            
        self._is_trained_ = True
        print("Alpha after training", self.alpha)

        if self.logger is not None:
            self.logger.info("Training done")


    def sample(self, nsamples, nchains=1, seed=None):
        """helper for the sample method from the Sampler class"""

       
        self._is_initialized() # check if the system is initialized


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

        sample_results ,  sampled_positions, local_energies = self.sampler.sample(nsamples, nchains)
        system_info = pd.DataFrame(system_info, index=[0])
        sample_results = pd.DataFrame(sample_results, index=[0])

    
        system_info_repeated = system_info.loc[
            system_info.index.repeat(len(sample_results))
        ].reset_index(drop=True)

    

        self._results = pd.concat([system_info_repeated, sample_results], axis=1)

        #print("this is the sampled position" , sampled_positions)

        local_energies = self.backend.array(local_energies)
        sampled_positions = self.backend.array(sampled_positions)

        return self._results, sampled_positions, local_energies
    

    def bootstrap(self,X):
        """"Here we write the bootstrap method, should take in the array of local energies calculated from the 
            sample member function of quantum states.

            Input:(X) 1D array of elements 

            Return:(X_boot_mean) retunrns the mean of the bootstrapped input array   
        """
        
        self.X = X
        nstraps = len(X)

        X_strap = np.random.choice(X,size = nstraps)
        
        
        return np.mean(X_strap)
    
    def superBoot(self,X,n):
        """
            This function takes n bootstraps and takes the mean of the output of all bootstraps

            Input: (X) 1D array of elements we wish to bootstrap
                   (n) number of times we wish to bootstrap
            
            Return: mean of all bootstrap means
        """
        self._n = n
        self._X = X
        meanE = []

        for _ in range(n):
            meanE.append(self.bootstrap(self._X))
        
        return np.mean(meanE)

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