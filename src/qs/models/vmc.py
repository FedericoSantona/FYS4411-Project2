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
        self._n_hidden= h_number,
        self.radius = config.radius,
        self._Nvis = self._dim*self._N,
        
        self.rng = random.PRNGKey(self._seed)  # Initialize RNG with the provided seed

    
        self._initialize_variational_params()
        

        self.state = 0  # take a look at the qs.utils State class
        self._initialize_vars(nparticles, dim, log, logger, logger_level)

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

        if config.interaction == "Coulomb" or config.hamiltonian == "eo":
            print("Coulomb interaction")
            self.wf_closure = self.wf_closure_int

        if config.interaction == "Coulomb" and config.hamiltonian == "eo":
            self.laplacian_closure = self.anal_laplacian_closure_int

            if backend != "jax":
                raise ValueError(
                    f"Backend {self.backend} not supported for Coulomb interaction"
                )

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
        a = self.params.get("a")  # Using Parameter.get to access alpha
        b = self.params.get("b")
        W = self.params.get("W")

        return self.wf_closure(r, a, b, W)

    def wf_closure_train(self, r, alpha):
        """

        r: (N, dim) array so that r_i is a dim-dimensional vector
        alpha: (N, dim) array so that alpha_i is a dim-dimensional vector

        return: should return Ψ(alpha, r)

        OBS: We strongly recommend you work with the wavefunction in log domain.

        """
        wf = -alpha * (
            self.beta**2 * (r[:, 0] ** 2) + self.backend.sum(r[:, 1:] ** 2, axis=1)
        )

        return wf

    def wf_closure(self, r, a, b, W):
        """

        r: (N, dim) array so that r_i is a dim-dimensional vector
        alpha: (N, dim) array so that alpha_i is a dim-dimensional vector

        return: should return Ψ(alpha, r)

        OBS: We strongly recommend you work with the wavefunction in log domain.

        """
        first_sum =  self.backend.sum((r-a)**2, axis = 1) /4
        lntrm = 1+self.backend.exp(b+self.backend.sum(self.backend.sum(r*W[None,:,:],axis = 2), axis = 1))
        second_sum = self.backend.sum(lntrm)/2
        wf = -first_sum + second_sum
        
        return wf

    def wf_closure_int(self, r, alpha):
        """

        r: (N, dim) array so that r_i is a dim-dimensional vector
        alpha: (1,1) array so that alpha is just a number but in the array form

        return: should return a an array of the wavefunction for each particle ( N, )

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """

        g = -alpha * (
            self.beta**2 * (r[:, 0] ** 2) + self.backend.sum(r[:, 1:] ** 2, axis=1)
        )  # Sum over the coordinates x^2 + y^2 + z^2 for each particle
        # Calculate pairwise distances.
        distances = self.la.norm(self.state.r_dist, axis=-1)
        # Compute f using the masked distances
        f = jnp.log(
            jnp.where(distances < self.radius, 0, 1 - self.radius / distances)
            + jnp.eye(r.shape[0])
        )
        f_term = self.backend.sum(f, axis=1)
        wf = g + f_term

        return wf

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
        grad_log_psi = jax.grad(
            lambda positions: jnp.sum(self.wf_closure(positions, alpha)), argnums=0
        )

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
        grad_alpha = self.backend.sum(
            -self.backend.sum(r**2, axis=2), axis=1
        )  # The gradient with respect to alpha

        return grad_alpha

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

        alpha = self.params.get("alpha")  # Using Parameter.get to access alpha
        return self.laplacian_closure(r, alpha)

    def laplacian_closure(self, r, alpha):
        """
        Analytical expression for the laplacian of the wavefunction
        """
        # For a Gaussian wavefunction in log domain, the Laplacian is simply 2*alpha*d - 4*alpha^2*sum(r_i^2),
        # where d is the number of dimensions.
        
        r = r.reshape(-1, self._dim)  # Reshape to (1 , n_dimensions)
        d = r.shape[1]  # Number of dimensions
        r2 = self.backend.sum(r**2, axis=1)

        laplacian = (
            4 * alpha**2 * r2 - 2 * alpha * d
        )  
        laplacian = self.backend.sum(
            laplacian.reshape(-1, 1)
        )  # Reshape to (n_particles, 1)

        return laplacian

    def uprime(self, rij):
        return self.radius / (rij**2 - self.radius * rij)

    def u_double_prime(self, rij):
        return (
            self.radius * (self.radius - 2 * rij) / ((rij**2 - self.radius * rij) ** 2)
        )

    def anal_laplacian_closure_int(self, r, alpha):
        """Something something soon easter boys - this should take osme arguments and return some value?"""

        k_indx = np.where(self.state.positions == r)[0][0]
        k_distances = self.la.norm(self.state.r_dist + 1e-8, axis=-1)[k_indx]
        self.k_distances = jnp.delete(k_distances, k_indx, axis=0)
        r_dist = self.state.positions - r + 1e-8
        r_dist = jnp.delete(r_dist, k_indx, axis=0)
        x, y, z = r
        self.grad_phi = -2 * alpha * jnp.array([x, y, self.beta * z])
        self.grad_phi_square = 4 * alpha**2 * (
            x**2 + y**2 + self.beta**2 * z**2
        ) - 2 * alpha * (self.beta + 2)

        self.first_term = self.grad_phi_square

        self.second_term_sum = jnp.sum(
            (r_dist.T / self.k_distances * self.uprime(self.k_distances)).T, axis=0
        )  # sum downwards, to keep it as a vector

        self.second_term = 2 * jnp.dot(self.grad_phi, self.second_term_sum)
        self.third_term = jnp.dot(self.second_term_sum, self.second_term_sum)
        self.fourth_term = jnp.sum(
            self.u_double_prime(self.k_distances)
            + 2 / self.k_distances * self.uprime(self.k_distances)
        )

        self.anal_laplacian = (
            self.first_term + self.second_term + self.third_term + self.fourth_term
        )

        return self.anal_laplacian

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
        a = self.params.get("alpha")  # Using Parameter.get to access alpha
        initial_logp = 0  # self.prob_closure(initial_positions , a)  # Now I use the log of the modulus of wave function, can be changed

        self.state = State(
            positions=initial_positions, logp=initial_logp, n_accepted=0, delta=0
        )
        self.state.r_dist = initial_positions[None, ...] - initial_positions[:, None, :]

    def _initialize_variational_params(self):
        # Initialize variational parameters in the correct range with the correct shape
        # Take a look at the qs.utils.Parameter class. You may or may not use it depending on how you implement your code.
        
        # Initialize the Boltzmann machine parameters
        a =  np.random.normal(0,1,size = (self._N,self._dim) )
        b =  np.random.normal(0,1,size = self._n_hidden )
        W =  np.random.normal(0,1,size = (self._n_hidden, self._N, self._dim) )

        initial_params = {"a": self.backend.array(a),"b": self.backend.array(b),"W": self.backend.array(W)}
        
        self.params = Parameter(
            initial_params
        )  # I still do not understand what should be the alpha dim
        pass
