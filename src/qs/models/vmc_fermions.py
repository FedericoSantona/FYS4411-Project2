import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import device_get
from jax import lax
from jax import vmap
import itertools
from qs.utils import (
    Parameter,
)  # IMPORTANT: you may or may not use this depending on how you want to implement your code and especially your jax gradient implementation
from qs.utils import State
import pdb
from simulation_scripts import config


class VMC_fermions:
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
        
        self._max_degree = config.max_degree
        self.n_up , self.n_down = self.generate_quantum_states(self._N , self._dim)

        self.state = 0  # take a look at the qs.utils State class
        self._initialize_vars(nparticles, dim, log, logger, logger_level)

        if self.log:
            n_para = self._M + self._n_hidden + self._n_hidden*self._M
            msg = f"""VMC initialized with {self._N} particles in {self._dim} dimensions with {n_para} parameters"""
            self._logger.info(msg)


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
            "single_particle_wavefunction_closure",
            "slater_determinant",
        ]

        for func in functions_to_jit:
            setattr(self, func, jax.jit(getattr(self, func)))
        return self
    

    def generate_quantum_states(self, N_particles, n_dim):
        
        N = N_particles // 2 
        # Generate a reasonable range of quantum numbers for each dimension
        max_quantum_number = int(jnp.ceil(N ** (1/n_dim)))


        states = list(itertools.product(range(max_quantum_number), repeat=n_dim))
        
        # Sort states by the sum of their components (which corresponds to energy levels in an isotropic harmonic oscillator)
        states.sort(key=sum)

        states = states[:N]


        states_array = np.array(states)

        # I am filling up each level with 2 electrons before to move to the next level ##,
        # even though they have the same energy
        spin_up_states = np.copy(states_array)
        spin_down_states = np.copy(states_array)

        return spin_up_states ,spin_down_states



    def hermite_poly(self, max_degree, x, active_degree):

        def scan_fun(carry, i):
            H_n_minus_2, H_n_minus_1 = carry
            H_n = 2 * x * H_n_minus_1 - 2 * (i - 1) * H_n_minus_2
            return (H_n_minus_1, H_n), H_n

        H_n_minus_2 = jnp.ones_like(x)
        H_n_minus_1 = 2 * x

        _, scanned_results = lax.scan(scan_fun, (H_n_minus_2, H_n_minus_1), jnp.arange(2, max_degree + 1))

        results = jnp.concatenate([jnp.array([H_n_minus_2, H_n_minus_1]), scanned_results])

        selected_result = results[active_degree]
        return selected_result
    




    def single_particle_wavefunction(self, r, n):
        """
        Helper for the single particle wave function

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """

        return self.single_particle_wavefunction_closure(r, n )

    
    def single_particle_wavefunction_closure(self, r, n):
        """
        
        In the log domain
        
        """
        scaled_r =r #jnp.sqrt(2 * alpha) * r
        n_ = n

        # Map over each quantum number
        hermite_values = vmap(self.hermite_poly, in_axes=(None, 0, 0))(self._max_degree, scaled_r, n)
        
        prod_hermite = jnp.prod(hermite_values)

        wf = prod_hermite #* jnp.exp(-alpha * jnp.sum(r ** 2))

        
        return wf.squeeze()
    

    def slater_determinant(self , r ):

        N = self._N// 2

        r_up = r[:N]  # Contains the first half of the particles
        r_down = r[N:]  # Contains the second half

        n_up = self.n_up
        n_down = self.n_down


        outer_vmap = vmap(lambda single_r, n: vmap(lambda single_n: self.single_particle_wavefunction_closure(single_r, single_n))(n), in_axes=(0, None))

        D_up = outer_vmap(r_up, n_up).reshape( N , N ).T
        D_down = outer_vmap(r_down, n_down).reshape( N , N ).T #now they are how you would expect them to be

        slater_det_up = jnp.linalg.det(D_up)
        slater_det_down =jnp.linalg.det(D_down)

        return slater_det_up * slater_det_down



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
        #Is 0.5 here because I am using the  psi = F
        first_term = self.backend.exp(- 0.5 * self.backend.sum((r_flat-a)**2) )

        jastrow = 1+self.backend.exp(b+self.backend.sum(r_flat[:, None]*W , axis = 0))

        second_term =  self.backend.prod(jastrow)
        
        boltzmann = first_term * second_term

        slater_det = self.slater_determinant(r)

       
        wf = boltzmann * slater_det

        
        return  wf
    
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

        return (self.wf_closure(r , a , b, W ))**2 # Since we're not in the log domain

    

    def grad_wf(self, r):
        """
        Helper for the gradient of the wavefunction with respect to r

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """
        a = self.params.get("a")  
        b = self.params.get("b")
        W = self.params.get("W")  

        return self.grad_wf_closure(r, a , b, W)


    def grad_wf_closure(self, r, a,b,W):
        """
        computes the gradient of the wavefunction with respect to r, but with jax grad
        """

        # Now we use jax.grad to compute the gradient with respect to the first argument (r)
        # Note: jax.grad expects a scalar output, so we sum over the particles to get a single value.
        grad_psi = jax.grad(
            lambda positions: jnp.sum(self.wf_closure(positions, a,b,W)), argnums=0
        )

        grad = grad_psi(r) / self.wf_closure(r, a,b,W)

        

        return grad.squeeze()



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
        Computes the gradient of the wavefunction with respect to the variational parameters with JAX grad using Vmap.
        """
        
        # Define a helper function to compute the gradient and divide by the wavefunction
        def grad_ratio(param_idx, wf_closure, r, a, b, W):
            grad_wf = jax.grad(wf_closure, param_idx)  # Compute gradient with respect to one parameter
            wf_value = wf_closure(r, a, b, W)          # Compute the wavefunction value
            return grad_wf(r, a, b, W) / wf_value      # Return the ratio of gradient to wavefunction value

        # Vectorize the helper function for each parameter
        grad_a_func = lambda r, a, b, W: grad_ratio(1, self.wf_closure, r, a, b, W)
        grad_b_func = lambda r, a, b, W: grad_ratio(2, self.wf_closure, r, a, b, W)
        grad_W_func = lambda r, a, b, W: grad_ratio(3, self.wf_closure, r, a, b, W)
        # Apply vectorization over the batch
        grad_a_batch = jax.vmap(grad_a_func, (0, None, None, None), 0)(r, a, b, W)
        grad_b_batch = jax.vmap(grad_b_func, (0, None, None, None), 0)(r, a, b, W)
        grad_W_batch = jax.vmap(grad_W_func, (0, None, None, None), 0)(r, a, b, W).reshape(self.batch_size,self._M*self._n_hidden)


        return grad_a_batch, grad_b_batch, grad_W_batch

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
        Computes the Laplacian of the wavefunction for each particle using JAX automatic differentiation.
        r: Position array of shape (n_particles, n_dimensions)
        alpha: Parameter(s) of the wavefunction
        """
        # Compute the Hessian (second derivative matrix) of the wavefunction
        hessian_psi = jax.hessian(
           lambda positions:  self.wf_closure(positions ,a ,b ,W), argnums=0
        )  # The hessian matrix for our wavefunction
        
        diagonal = jnp.diag(
            hessian_psi(r).reshape(self._dim*self._N , self._dim*self._N)
        ) # The hessian is nested like a ... onion

        
        laplacian = jnp.sum(diagonal) / self.wf_closure(r,a,b,W) # VERY UNSURE ABOUT THIS


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
