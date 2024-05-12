import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from qs.models.vmc import VMC
from qs.utils import Parameter
from simulation_scripts import config
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import time
class Hamiltonian:
    def __init__(
        self,
        nparticles,
        dim,
        int_type,
        backend,
    ):
        """
        Note that this assumes that the wavefunction form is in the log domain
        """
        self._N = nparticles
        self._dim = dim
        self._int_type = config.interaction

        match backend:
            case "numpy":
                self.backend = np
                self.la = np.linalg
            case  "jax":
                self.backend = jnp
                self.la = jnp.linalg
                
            case _: # noqa
                raise ValueError("Invalid backend:", backend)

    def local_energy(self, wf, r):
        """Local energy of the system"""
        raise NotImplementedError



class HarmonicOscillator(Hamiltonian):
    def __init__(
        self,
        alg_int,
        nparticles,
        dim,
        log,
        logger,
        seed,
        logger_level,
        int_type,
        backend,
    ):
        # Initialize the parent class, which sets up the backend among other things
        super().__init__(nparticles, dim, int_type, backend)
        
        # Set additional attributes specific to HarmonicOscillator
        self.seed = seed
        self.log = log
        self.logger = logger
        self.logger_level = logger_level
        self.alg_int = alg_int


    # I dont think we should use JAX here either - local energy is called repeatedly, and thus not 
    # easily compiled in JAX. The JNP.arrays should be used _only_ where we can actually run the
    # JAX JIT compiler. I think it'll potentially reduce the performance of the program otherwise.
    
    def CoulombInteraction(self, r):
        r_copy = r.copy()
        a, b = self.backend.triu_indices(r_copy[:,0].size,1)
        r_dist = self.la.norm(r_copy[a] - r_copy[b], axis=1)
        #r_dist = self.la.norm(r_copy[None, ...] - r_copy[:, None, :], axis=-1)
        #r_dist = self.backend.where(r_dist < config.radius, 0, r_dist)     
        #breakpoint()
        int_energy = self.backend.sum(
            #self.backend.triu(1 / r_dist, k=1)
            1/r_dist
        )   # Calculates the upper triangular of the distance matrix (to not do a double sum)

        return int_energy
    

    def non_int_energy(self, r):
        omega = config.omega
        
        kinetic_energy = self.alg_int.laplacian(r)
        potential_energy = self.backend.sum(omega**2*(r**2).flatten())

        #The sum without specific axis is the sum of all elements in the array i.e. returns a scalar
        non_int_energy =  0.5*(-kinetic_energy + potential_energy) 
        #print(non_int_energy)
        return non_int_energy


    def local_energy(self,r):

        non_int_energy = self.non_int_energy(r)
        interaction_energy = 0

        if self._int_type == "Coulomb":
            interaction_energy = self.CoulombInteraction(r)
        else:
            pass

        return non_int_energy + interaction_energy

