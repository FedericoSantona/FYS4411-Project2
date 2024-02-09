import jax
import jax.numpy as jnp
import numpy as np
from qs.models.vmc import VMC

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
        self._int_type = int_type

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
        self._N = nparticles
        self._dim = dim
        self._int_type = int_type
        self.backend = backend
        self.seed = seed
        self.log = log
        self.logger = logger
        self.logger_level = logger_level
        self.alg_int = alg_int

       
        super().__init__(nparticles, dim, int_type, backend)

    def local_energy(self, wf, r):
       
        """Local energy of the system
        Calculates the local energy of a system with positions `r` and wavefunction `wf`.
        `wf` is assumed to be the log of the wave function.
        """
        # Potential Energy
        pe = 0.5 * jnp.sum(r**2, axis=1).mean()  # Sum over dimensions, mean over particles

        # Kinetic Energy using automatic differentiation on the log of the wavefunction
        laplacian = jnp.sum(self.alg_int.laplacian(r) , axis= 1).mean()

        # Correct calculation of local energy
        local_energy = -0.5 * laplacian + pe
        

    
        return local_energy