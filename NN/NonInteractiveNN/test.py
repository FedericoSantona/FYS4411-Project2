import itertools
import numpy as np
from numba import njit
import jax.numpy as jnp
from jax import lax 
from jax import vmap, jit

def generate_quantum_states(N_particles, n_dim):
    N = N_particles // 2 
    max_quantum_number = int(np.ceil(N ** (1 / n_dim)))
    states = list(itertools.product(range(max_quantum_number), repeat=n_dim))
    states.sort(key=sum)
    states = states[:N]
    states_array = np.array(states)

    spin_up_states = np.copy(states_array)
    spin_down_states = np.copy(states_array)

    return spin_up_states, spin_down_states

@njit
def hermite_poly_numba(max_degree, x, active_degree):
    H_n_minus_2 = np.ones_like(x)
    H_n_minus_1 = 2 * x

    results = [H_n_minus_2, H_n_minus_1]
    for i in range(2, max_degree + 1):
        H_n = 2 * x * H_n_minus_1 - 2 * (i - 1) * H_n_minus_2
        results.append(H_n)
        H_n_minus_2, H_n_minus_1 = H_n_minus_1, H_n

    results = np.array(results)
    selected_result = results[active_degree]
    return np.abs(selected_result)

@njit
def single_particle_wavefunction_closure_numba(r, n):
    scaled_r = r
    hermite_values = np.array([hermite_poly_numba(10, scaled_r[i], n[i]) for i in range(len(n))])
    prod_hermite = np.prod(hermite_values)
    wf = np.log(prod_hermite)
    return wf

@njit
def outer_vmap_numba(r, n, N):
    result = np.empty((N, N))
    for i in range(len(r)):
        result[i] = single_particle_wavefunction_closure_numba(r[i], n)
    return result

@njit
def slater_determinant_numba(r, n_up, n_down, N_particles):
    N = N_particles // 2
    r_up = r[:N]
    r_down = r[N:]

    D_up = outer_vmap_numba(r_up, n_up, N).reshape(N, N).T
    D_down = outer_vmap_numba(r_down, n_down, N).reshape(N, N).T

    slater_det_up = np.linalg.det(D_up)
    slater_det_down = np.linalg.det(D_down)

    return slater_det_up * slater_det_down


def hermite_poly_jax( max_degree, x, active_degree):

    def scan_fun(carry, i):
        H_n_minus_2, H_n_minus_1 = carry
        H_n = 2 * x * H_n_minus_1 - 2 * (i - 1) * H_n_minus_2
        return (H_n_minus_1, H_n), H_n

    H_n_minus_2 = jnp.ones_like(x)
    H_n_minus_1 = 2 * x

    _, scanned_results = lax.scan(scan_fun, (H_n_minus_2, H_n_minus_1), jnp.arange(2, max_degree + 1))

    results = jnp.concatenate([jnp.array([H_n_minus_2, H_n_minus_1]), scanned_results])

    selected_result = results[active_degree]
    return jnp.abs(selected_result)

def slater_determinant_jax(r, n_up, n_down, N_particles):
    

    @jit
    def single_particle_wavefunction_closure_jax(r, n):
        scaled_r = r
        hermite_values = vmap(hermite_poly_jax, in_axes=(None, 0, 0))(10, scaled_r, n)
        prod_hermite = jnp.prod(hermite_values)
        wf = jnp.log(prod_hermite)
        return wf

    N = N_particles // 2
    r_up = r[:N]
    r_down = r[N:]

    outer_vmap = vmap(lambda single_r, n: vmap(lambda single_n: single_particle_wavefunction_closure_jax(single_r, single_n))(n), in_axes=(0, None))
    D_up = outer_vmap(r_up, n_up).reshape(N, N).T
    D_down = outer_vmap(r_down, n_down).reshape(N, N).T

    slater_det_up = jnp.linalg.det(D_up)
    slater_det_down = jnp.linalg.det(D_down)

    return slater_det_up * slater_det_down

# Example usage and comparison
N_particles = 4
n_dim = 2
r = np.random.rand(N_particles, n_dim)
n_up, n_down = generate_quantum_states(N_particles, n_dim)

# JAX version
slater_det_jax = slater_determinant_jax(r, n_up, n_down, N_particles)

# Numba version
slater_det_numba = slater_determinant_numba(r, n_up, n_down, N_particles)

print("JAX Slater Determinant:", slater_det_jax)
print("Numba Slater Determinant:", slater_det_numba)
print("Are the results close?", np.allclose(slater_det_jax, slater_det_numba))
