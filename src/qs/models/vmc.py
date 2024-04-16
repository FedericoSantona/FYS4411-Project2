import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import device_get
from qs.utils import (
    Parameter,
)  # IMPORTANT: you may or may not use this depending on how you want to implement your code and especially your jax gradient implementation
from qs.utils import State
import pdb
from simulation_scripts import config


class VMC:
    def __init__(
        self,
        nparticles,
        dim,
        h_number,
        rng=None,
        log=False,
        logger=None,
        seed=None,
        logger_level="INFO",
        backend="numpy",
        radius=None,
    ):
        self._configure_backend(backend)
        self.params = Parameter()
        self.log = log
        self._seed = seed
        self._logger = logger
        self._N = nparticles
        self._dim = dim
        self._M = self._N * self._dim
       
        self._n_hidden = h_number

        self.radius = config.radius
        self.batch_size = config.batch_size
        
        
        
        self.rng = random.PRNGKey(self._seed)  # Initialize RNG with the provided seed

    
        self._initialize_variational_params()
        

        self.state = 0  # take a look at the qs.utils State class
        self._initialize_vars(nparticles, dim, log, logger, logger_level)

        if self.log:
            n_para = self._M + self._n_hidden + self._n_hidden*self._M
            msg = f"""VMC initialized with {self._N} particles in {self._dim} dimensions with {n_para} parameters"""
            self._logger.info(msg)

    def generate_normal_matrix(self, n, m):
        """Generate a matrix of normally distributed numbers with shape (n, m)."""
        if self.rng is None:
            raise ValueError("RNG key not initialized. Call set_rng first.")

        # Split the key: one for generating numbers now, one for future use.
        self.rng, subkey = random.split(self.rng)

        # Generate the normally distributed numbers
        return random.normal(subkey, shape=(n, m))

    def _configure_backend(self, backend):
        """
        Here we configure the backend, for example, numpy or jax
        You can use this to change the linear algebra methods or do just in time compilation.
        Note however that depending on how you build your code, you might not need this.
        """
        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg

            # Here we overwrite the functions with their jax versions. This is just a suggestion.
            # These are also the _only_ functions that should be written in JAX code, but should we then
            # convert back and forth from JAX <-> NumPy arrays throughout the program?
            # Need to discuss with Daniel.
            self.grad_wf_closure = self.grad_wf_closure_jax
            self.grads_closure = self.grads_closure_jax
            self.laplacian_closure = self.laplacian_closure_jax
            self._jit_functions()
        else:
            raise ValueError(f"Backend {self.backend} not supported")


    def _jit_functions(self):
        """
        Note there are other ways to jit functions. this is just one example.
        However, you should be careful with how you jit functions.
        They have to be pure functions, meaning they cannot have side effects (modify some state variable values outside its local environment)
        Take a close look at "https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html"
        """

        functions_to_jit = [
            "prob_closure",
            "wf_closure",
            "grad_wf_closure",
            "laplacian_closure",
            "grads_closure",
        ]

        for func in functions_to_jit:
            setattr(self, func, jax.jit(getattr(self, func)))
        return self

    def wf(self, r):
        """
        Helper for the wave function

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """
        a = self.params.get("a")  
        b = self.params.get("b")
        W = self.params.get("W")

        return self.wf_closure(r, a, b, W)

    

    def wf_closure(self, r, a, b, W):
        """

        r: (N, dim) array so that r_i is a dim-dimensional vector
        a: (N, dim) array so that a_i is a dim-dimensional vector
        b: (N, 1) array represents the number of hidden nodes
        W: (N_hidden, N , dim) array represents the weights


        OBS: We strongly recommend you work with the wavefunction in log domain.

        """

        r_flat = r.flatten()
        first_sum =  0.25 * self.backend.sum((r_flat-a)**2) 

        lntrm =self.backend.log( 1+self.backend.exp(b+self.backend.sum(r_flat[:, None]*W , axis = 0)))

        second_sum = 0.5 * self.backend.sum(lntrm)
        
        wf = -first_sum + second_sum
        
        return wf

    def prob(self, r):
        """
        Helper for the probability density

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """

        a = self.params.get("a")  
        b = self.params.get("b")
        W = self.params.get("W")
          
        return self.prob_closure(r, a , b, W)
    
    def prob_closure(self, r, a , b, W):
        """
        Return a function that computes |Ψ(alpha, r)|^2

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """

        log_psi = self.wf_closure(r, a , b , W)

        return 2 * log_psi  # Since we're working in the log domain
    

    def grad_wf(self, r):
        """
        Helper for the gradient of the wavefunction with respect to r

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """
        a = self.params.get("a")  
        b = self.params.get("b")
        W = self.params.get("W")  

        return self.grad_wf_closure(r, a , b, W)

    def grad_wf_closure(self, r, a , b, W):
        """
        Computes the gradient of the wavefunction with respect to r analytically
        Is overwritten by the JAX version if backend is JAX

        r: (N, dim) array so that r_i is a dim-dimensional vector
        a: (M ,) array so that a_i is a dim-dimensional vector
        b: (N_hidd, ) array represents the number of hidden nodes
        W: (M , N_hidd) array represents the weights


        Here the output will be of shape (M ,) because we are taking the gradient with respect to
        every visible nodes and conserve the informations
        """
        r_flat = r.flatten()

        first_term = 0.5 * (r_flat - a )

        exp_term =   1+self.backend.exp(-(b+self.backend.sum(r_flat[:, None]*W , axis = 0)))

        second_term = 0.5 * self.backend.sum(W / exp_term , axis = 1)

        """
        second_term1 = []
        for m in range(self._M):
            second_term1.append([])
            for n in range(self._n_hidden):
                second_term1[-1].append(W[m,n] / exp_term[n])
        second_term1 = 0.5*np.sum(np.array(second_term1), axis = 1)
        """

        grad = -first_term + second_term

        return grad


    def grad_wf_closure_jax(self, r, alpha):
        """
        computes the gradient of the wavefunction with respect to r, but with jax grad
        """

        # Now we use jax.grad to compute the gradient with respect to the first argument (r)
        # Note: jax.grad expects a scalar output, so we sum over the particles to get a single value.
        grad_log_psi = jax.grad(
            lambda positions: jnp.sum(self.wf_closure(positions, alpha)), argnums=0
        )

        return grad_log_psi(r)



    def grads(self, r):
        """
        Helper for the gradient of the wavefunction with respect to the variational parameters

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """
        a = self.params.get("a")  
        b = self.params.get("b")
        W = self.params.get("W") 

        return self.grads_closure(r, a , b, W)

    def grads_closure(self, r, a, b, W):
        """
        Computes the gradient of the wavefunction with respect to the variational parameters analytically

        Here r comes in shape (n_batch , n_particles , n_dim)
        """

        r_flat = r.reshape(self.batch_size , -1)


        #grad_a is of shape (N_batch , M)
        grad_a = (0.5 * (r_flat - a))


        # grad_b is of shape (n_batch , n_hidden)
        grad_b = 1 / (2*( 1+self.backend.exp(-(b+self.backend.sum(r_flat[:,: ,None]*W[None,:,:] , axis = 1)))))

        #grad_W is of shape (n_batch ,  M * N_hidden )
        
        grad_W  = (r_flat[:,:,None] * grad_b[:,None,:]).reshape(self.batch_size , self._M * self._n_hidden)
        

        return grad_a , grad_b , grad_W

    def grads_closure_jax(self, r, alpha):
        """
        Computes the gradient of the wavefunction with respect to the variational parameters with JAX grad using Vmap.
        """

        # Define the gradient function for a single instance
        def single_grad(pos, var):
            return jax.grad(lambda a: jnp.sum(self.wf_closure_train(pos, a)))(var)

        # Vectorize the gradient computation over the batch dimension
        batched_grad = jax.vmap(single_grad, (0, None), 0)

        # Compute gradients for the entire batch
        grad_alpha = batched_grad(r, alpha)

        return self.backend.squeeze(grad_alpha)

    def laplacian(self, r):
        """
        Return a function that computes the laplacian of the wavefunction ∇^2 Ψ(r)

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """

        a = self.params.get("a")  
        b = self.params.get("b")
        W = self.params.get("W")  

        return self.laplacian_closure(r, a , b , W)

    def laplacian_closure(self, r, a , b , W):
        """
        Analytical expression for the laplacian of the wavefunction
        Here the output is still of shape (M,) conserving the laplacian
        for each visible node
        """
        r_flat = r.flatten()
        
        num = self.backend.exp(b+self.backend.sum(r_flat[:, None]*W , axis = 0))
        den = (1+self.backend.exp(b+self.backend.sum(r_flat[:, None]*W , axis = 0)))**2

        term = num/den

        laplacian = -0.5 +0.5*self.backend.sum(W**2 * term, axis = 1)

        return laplacian

    
    def laplacian_closure_jax(self, r, alpha):
        """
        Computes the Laplacian of the wavefunction for each particle using JAX automatic differentiation.
        r: Position array of shape (n_particles, n_dimensions)
        alpha: Parameter(s) of the wavefunction
        """
        # Compute the Hessian (second derivative matrix) of the wavefunction
        hessian_psi = jax.hessian(
            self.wf, argnums=0
        )  # The hessian matrix for our wavefunction
        d = self._dim

        r = r.reshape(1, d)
        first_term = jnp.trace(
            hessian_psi(r)[0].reshape(d, d)
        )  # The hessian is nested like a ... onion
        second_term = jnp.sum(self.grad_wf_closure(r, alpha) ** 2)
        laplacian = first_term + second_term

        return laplacian

    def _initialize_vars(self, nparticles, dim, log, logger, logger_level):
        """Initializing the parameters in the VMC instance"""
        assert isinstance(nparticles, int), "nparticles must be an integer"
        assert isinstance(dim, int), "dim must be an integer"
        self._N = nparticles
        self._dim = dim
        self._log = log if log else False

        if logger:
            self._logger = logger
        else:
            import logging

            self._logger = logging.getLogger(__name__)

        self._logger_level = logger_level

        # Generate initial positions randomly
        # Note: We split the RNG key to ensure subsequent uses of RNG don't reuse the same state.
        key, subkey = random.split(self.rng)
        initial_positions = random.normal(
            subkey, (nparticles, dim)
        )  # Using JAX for random numbers
        # Initialize logp, assuming a starting value or computation
        
        initial_logp = 0  # self.prob_closure(initial_positions , a)  # Now I use the log of the modulus of wave function, can be changed

        self.state = State(
            positions=initial_positions, logp=initial_logp, n_accepted=0, delta=0
        )
        self.state.r_dist = initial_positions[None, ...] - initial_positions[:, None, :]

    def _initialize_variational_params(self):
        # Initialize variational parameters in the correct range with the correct shape
        # Take a look at the qs.utils.Parameter class. You may or may not use it depending on how you implement your code.
        
        # Initialize the Boltzmann machine parameters
        a =   np.random.normal(0,config.init_scale,size = self._M )
        b =  np.random.normal(0,config.init_scale,size = self._n_hidden )
        W =  np.random.normal(0,config.init_scale,size = (self._M , self._n_hidden) )

        
        initial_params = {"a": self.backend.array(a),"b": self.backend.array(b),"W": self.backend.array(W)}
        
        self.params = Parameter(
            initial_params
        )  # I still do not understand what should be the alpha dim
        pass
