import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from qs.models.vmc import VMC
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
                # self.kinetic_energy = jax.pmap(self.kinetic_energy) NB!! This should be looked at
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
    
    def kinetic_energy(self, r):
        """Kinetic energy of the system"""
        # I believe there should be a way to easily parallelize this for loop (i.e split it into smaller for-loops that run in parallel)
        laplacian = 0
        for i in range(self._N):
            laplacian += self.alg_int.laplacian(r[i])
            
        return -0.5 * laplacian
    
    def local_energy(self, wf, r):
        """Local energy of the system
        Calculates the local energy of a system with positions `r` and wavefunction `wf`.
        `wf` is assumed to be the log of the wave function.
        """
        # Potential Energy
        pe = 0.5 * self.backend.sum(self.backend.sum(r**2, axis=1))  # Use self.backend to support numpy/jax.numpy 
        # Kinetic energy
        ke = self.kinetic_energy(r)
        # Correct calculation of local energy
        local_energy = ke + pe

        return local_energy
    

class EllipticOscillator(HarmonicOscillator):
    def __init__(self, alg_int, nparticles, dim, log, logger, seed, logger_level, int_type, backend, beta):
        super().__init__(alg_int, nparticles, dim, log, logger, seed, logger_level, int_type, backend)

        self.beta = beta  # Store the ellipticity parameter

    def potential_energy(self, r):
        """Calculates the potential energy
        """
        pe = 0.5 * self.backend.sum(self.beta**2 * r[:, 0]**2 + self.backend.sum(r[:, 1:]**2, axis=1))
        int_energy = 0

        if self._int_type == "Coulomb":
            r_copy = r.copy()
            r_dist = self.la.norm(r_copy[None, ...] - r_copy[:, None, :], axis=-1)
            r_dist = self.backend.where(r_dist < config.radius, 0, r_dist)                 # Should be edited to be self.radius or something, not 0.0043
            int_energy = self.backend.sum(
                self.backend.triu(1 / r_dist, k=1)
            )   # Calculates the upper triangular of the distance matrix (to not do a double sum)
        else:
            pass


        return pe + int_energy
    
    def local_energy(self, wf, r):
        ###TODO Impliment local energy for EO

        """Local energy of the system
        Calculates the local energy of a system with positions `r` and wavefunction `wf`.
        `wf` is assumed to be the log of the wave function.
        """
        # Adjust the potential energy calculation for the elliptic oscillator
        # Assuming r is structured as [nparticles, dim], and the first column is x, second is y, and the third is z.
        pe = self.potential_energy(r)
        ke = self.kinetic_energy(r)
        
        # Kinetic Energy using automatic differentiation on the log of the wavefunction 
        #print(" laplacian shape ", self.backend.sum(self.alg_int.laplacian(r)).shape)
        #laplacian = self.backend.sum(self.alg_int.laplacian(r)) this was more efficient tho :(
        
        # laplacian = 0
        # for i in range(self._N):
        #     laplacian += self.backend.sum(self.alg_int.laplacian(r[i]))
        
        
        # Correct calculation of local energy
        local_energy =  ke + pe

        #local_energy = self.backend.array(local_energy)
        #local_energy = local_energy.reshape((1, 1))

        return local_energy