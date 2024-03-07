import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import device_get
from qs.utils import Parameter # IMPORTANT: you may or may not use this depending on how you want to implement your code and especially your jax gradient implementation
from qs.utils import State
import pdb

class VMC:
    def __init__(
        self,
        nparticles,
        dim,
        rng=None,
        log=False,
        logger=None, 
        seed= None,
        logger_level="INFO",
        backend="numpy",
        alpha = None,
        
    ):
        
        self._configure_backend(backend)
        self.params = Parameter()
        self.log = log
        self._seed = seed
        self._logger = logger
        self._N = nparticles             
        self.rng = random.PRNGKey(self._seed)  # Initialize RNG with the provided seed
        
        if alpha:
            self._initialize_variational_params(alpha)
        else:
            self._initialize_variational_params() # initialize the variational parameters (ALPHA)

        self._initialize_vars(nparticles, dim, log, logger, logger_level)



        r = 0 # I do not know what is this
        
        self.state = 0 # take a look at the qs.utils State class

        if self.log:
            msg = f"""VMC initialized with {self._N} particles in {self._dim} dimensions with {
                    self.params.get("alpha").size
                    } parameters"""
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
                # "laplacian_closure",
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
        alpha = self.params.get("alpha")  # Using Parameter.get to access alpha
        
        return self.wf_closure(r, alpha)


    
    def wf_closure(self, r, alpha):
        """
        
        r: (N, dim) array so that r_i is a dim-dimensional vector
        alpha: (N, dim) array so that alpha_i is a dim-dimensional vector

        return: should return Ψ(alpha, r)

        OBS: We strongly recommend you work with the wavefunction in log domain. 
        """
        
        return -alpha * self.backend.sum(r**2, axis=1)  # Sum over the coordinates x^2 + y^2 + z^2 for each particle
    

    def prob_closure(self, r, alpha):
        """
        Return a function that computes |Ψ(alpha, r)|^2

        OBS: We strongly recommend you work with the wavefunction in log domain. 
        """
        
        log_psi = self.wf_closure(r, alpha)
        return 2 * log_psi  # Since we're working in the log domain


    def prob(self, r):
        """
        Helper for the probability density

        OBS: We strongly recommend you work with the wavefunction in log domain. 
        """
        alpha = self.params.get("alpha")  # Using Parameter.get to access alpha 
        return self.prob_closure(r, alpha)

    def grad_wf_closure(self, r, alpha):
        """
        Computes the gradient of the wavefunction with respect to r analytically
        Is overwritten by the JAX version if backend is JAX
        """

        # The gradient of the Gaussian wavefunction is straightforward:
        # d/dr -alpha * r^2 = -2 * alpha * r
        # We apply this to all dimensions (assuming r is of shape (n_particles, n_dimensions))
        return -2 * alpha * r
        

    def grad_wf_closure_jax(self, r, alpha):
        """
        computes the gradient of the wavefunction with respect to r, but with jax grad
        """
    
        # Now we use jax.grad to compute the gradient with respect to the first argument (r)
        # Note: jax.grad expects a scalar output, so we sum over the particles to get a single value.
        grad_log_psi = jax.grad(lambda positions: jnp.sum(self.wf_closure(positions, alpha)), argnums=0)

        return grad_log_psi(r)
        

    def grad_wf(self, r):
        """
        Helper for the gradient of the wavefunction with respect to r

        OBS: We strongly recommend you work with the wavefunction in log domain. 
        """
        alpha = self.params.get("alpha")  # Using Parameter.get to access alpha 
        
        return self.grad_wf_closure(r, alpha)

    def grads(self, r):
        """
        Helper for the gradient of the wavefunction with respect to the variational parameters

        OBS: We strongly recommend you work with the wavefunction in log domain. 
        """
        alpha = self.params.get("alpha")  # Using Parameter.get to access alpha 
   

        return self.grads_closure(r, alpha)

    def grads_closure(self, r, alpha):
        """
        Computes the gradient of the wavefunction with respect to the variational parameters analytically
        """
    
        # For the given trial wavefunction, the gradient with respect to alpha is the negative of the wavefunction
        # times the sum of the squares of the positions, since the wavefunction is exp(-alpha * sum(r_i^2)).
        grad_alpha = -self.backend.sum(r**2, axis=1)  # The gradient with respect to alpha
        return grad_alpha

    def grads_closure_jax(self, r, alpha):
        """
        Computes the gradient of the wavefunction with respect to the variational parameters with JAX grad.
        """
        # Assuming self.wf expects alpha to be a JAX array
        grad_alpha_fn = jax.grad(lambda a: jnp.sum(self.wf_closure(r, a)), argnums=0)
        grad_alpha = grad_alpha_fn(alpha)
        return grad_alpha

    def laplacian(self, r):
        """
        Return a function that computes the laplacian of the wavefunction ∇^2 Ψ(r)

        OBS: We strongly recommend you work with the wavefunction in log domain. 
        """
       
        alpha = self.params.get("alpha")  # Using Parameter.get to access alpha 
        
        return self.laplacian_closure(r, alpha)

    def laplacian_closure(self, r, alpha):
        """
        Analytical expression for the laplacian of the wavefunction
        """
        # print("ANALITICAL")
        # For a Gaussian wavefunction in log domain, the Laplacian is simply 2*alpha*d - 4*alpha^2*sum(r_i^2),
        # where d is the number of dimensions.
        r = r.reshape(-1, self._dim)  # Reshape to (n_particles, n_dimensions)
        d = r.shape[1]  # Number of dimensions
        r2 =  self.backend.sum(r**2, axis=1)
        
        laplacian = 4* alpha**2 * r2 - 2 * alpha * d # is this the actual log domain? what is the log domain?

        laplacian = laplacian.reshape(-1, 1)  # Reshape to (n_particles, 1)

        return laplacian

    def laplacian_closure_jax(self, r, alpha):
      
        """
        Computes the Laplacian of the wavefunction for each particle using JAX automatic differentiation.
        r: Position array of shape (n_particles, n_dimensions)
        alpha: Parameter(s) of the wavefunction
        """
        # Compute the Hessian (second derivative matrix) of the wavefunction
        hessian_psi = jax.hessian(self.wf, argnums=0)
        d = r.shape[1]
        r  =r.reshape(self._N,d)
        # Apply the Hessian function to the positions to get the second derivatives
        # And calculate the second term (since we are working in the log domain, we square the gradient of the wavefunction)
        # WE SHOULD USE .RAVEL() TO FIX THIS SHIT
        laplacian = 0
        #breakpoint()
        L = jax.lax.fori_loop(0, self._N, lambda i, laplacian: laplacian + 
                              jnp.trace(hessian_psi(jnp.array([r[i]])).reshape(d, d)) + jnp.sum(self.grad_wf_closure(jnp.array([r[i]]), alpha) ** 2), 0
                              ) 
        
        # first_term = hessian_psi(r).reshape(d[0], d[0]) # Reshape to (n_particles, n_dimensions, n_dimensions)
        # second_term = self.grad_wf_closure(r, alpha) ** 2

        # #sum the diagonal elements of the Hessian + the to get the Laplacian
        # laplacian = jnp.array(jnp.trace(first_term) + jnp.sum(second_term))
        breakpoint()
        return L
        return laplacian

    def _initialize_vars(self, nparticles, dim, log, logger, logger_level):
        
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
        initial_positions = random.normal(subkey, (nparticles, dim))  # Using JAX for random numbers

        # Initialize logp, assuming a starting value or computation
        a = self.params.get("alpha")  # Using Parameter.get to access alpha
        initial_logp = self.prob_closure(initial_positions , a)  # Now I use the log of the modulus of wave function, can be changed

        self.state = State(positions=initial_positions, logp=initial_logp , n_accepted= 0 , delta = 0)

    def _initialize_variational_params(self , alpha = None ):
        # Initialize variational parameters in the correct range with the correct shape
        # Take a look at the qs.utils.Parameter class. You may or may not use it depending on how you implement your code.
        # Here, we initialize the variational parameter 'alpha'.
        if alpha:
            initial_params = {"alpha": jnp.array([alpha])}
        else:
            initial_params = {"alpha": jnp.array([0.5])}  # Example initial value for alpha ( 1 paramter)
        self.params = Parameter(initial_params)  # I still do not understand what should be the alpha dim
        pass 