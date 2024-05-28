# import njit from numba
from numba import njit
# import np
import numpy as np

# initialize params with Dict of numba

from numba import types
from numba.typed import Dict

# randomseed
np.random.seed(0)

# import plt
import matplotlib.pyplot as plt

# import sqrt
from math import sqrt



####################### PARAMS UTILITIES



# define a function that help to produce params from the number of particles and dimension
# the hidden number of neuron
def init_params_rand(N, D, H):
    params = Dict.empty(
        key_type=types.unicode_type,
        # value type can be 1D or 2D array
        value_type = types.float64[:,:]
    )
    params['W0'] = np.random.randn(1, N)

    params['W1'] = np.random.randn(H, int(N*(N-1)/2))
    params['b1'] = np.random.randn(H,1)

    params['W2'] = np.random.randn(1, H)
    params['b2'] = np.random.randn(1,1)

    params['gamma'] = np.random.randn(1,1)

    return params

# init_params_W0_ones and all the rest is zero
def init_params_W0_ones(N, D, H):
    params = Dict.empty(
        key_type=types.unicode_type,
        # value type can be 1D or 2D array
        value_type = types.float64[:,:]
    )
    params['W0'] = -np.ones((1, N))

    params['W1'] = np.zeros((H, int(N*(N-1)/2)))
    params['b1'] = np.zeros((H,1))

    params['W2'] = np.zeros((1, H))
    params['b2'] = np.zeros((1,1))

    params['gamma'] = np.zeros((1,1))

    return params


def init_params_zeros(N, D, H):
    params = Dict.empty(
        key_type=types.unicode_type,
        # value type can be 1D or 2D array
        value_type = types.float64[:,:]
    )
    params['W0'] = np.zeros((1, N))

    params['W1'] = np.zeros((H, int(N*(N-1)/2)))
    params['b1'] = np.zeros((H,1))

    params['W2'] = np.zeros((1, H))
    params['b2'] = np.zeros((1,1))

    params['gamma'] = np.zeros((1,1)) + 1

    return params



# def a function that sum two params dictionary
@njit
def sum_params(params1, params2):
    params = Dict.empty(
        key_type=types.unicode_type,
        # value type can be 1D or 2D array
        value_type = types.float64[:,:]
    )
    for key in params1:
        params[key] = params1[key] + params2[key]
    return params


# def a func that multiply a params dictionary by a scalar
@njit
def scalar_mult_params(params, scalar):
    params_new = Dict.empty(
        key_type=types.unicode_type,
        # value type can be 1D or 2D array
        value_type = types.float64[:,:]
    )
    for key in params:
        params_new[key] = scalar*params[key]
    return params_new


# def set_gamma_params
@njit
def set_gamma_params(params, gamma):
    params_no_gamma = Dict.empty(
        key_type=types.unicode_type,
        # value type can be 1D or 2D array
        value_type = types.float64[:,:]
    )

    for key in params:
        if key != 'gamma':
            params_no_gamma[key] = params[key]
        else:
            params_no_gamma[key] = gamma*np.ones(params[key].shape)
    
    return params_no_gamma


# def init_params_Wh_rand is W1, W2, b1, b2 and the rest zero, control with epsilon
def init_params_Wh_rand(N, D, H, epsilon=0.1):
    params = Dict.empty(
        key_type=types.unicode_type,
        # value type can be 1D or 2D array
        value_type = types.float64[:,:]
    )
    params['W0'] = np.zeros((1, N))

    params['W1'] = epsilon*np.random.randn(H, int(N*(N-1)/2))
    params['b1'] = epsilon*np.random.randn(H,1)

    params['W2'] = epsilon*np.random.randn(1, H)
    params['b2'] = epsilon*np.random.randn(1,1)

    params['gamma'] = epsilon*np.random.randn(1,1)

    return params


# def_init_params_gamma_ones, all zero except gamma
def init_params_gamma_one(N, D, H):
    params = Dict.empty(
        key_type=types.unicode_type,
        # value type can be 1D or 2D array
        value_type = types.float64[:,:]
    )
    params['W0'] = np.zeros((1, N))

    params['W1'] = np.zeros((H, int(N*(N-1)/2)))
    params['b1'] = np.zeros((H,1))

    params['W2'] = np.zeros((1, H))
    params['b2'] = np.zeros((1,1))

    params['gamma'] = np.ones((1,1))

    return params


# def put the b coef in params to zero
def params_no_b(params):
    params_no_b = Dict.empty(
        key_type=types.unicode_type,
        # value type can be 1D or 2D array
        value_type = types.float64[:,:]
    )
    for key in params:
        if key != 'b1' and key != 'b2':
            params_no_b[key] = params[key]
        else :
            params_no_b[key] = np.zeros(params[key].shape)
    return params_no_b




# def_init_params_Guess, where W0 is alpha, Wh are random with epsilon, and gamma = gamma
def init_params_guess(N, D, H, alpha, epsilon=0.1, gamma=1):
    params = Dict.empty(
        key_type=types.unicode_type,
        # value type can be 1D or 2D array
        value_type = types.float64[:,:]
    )
    params['W0'] = alpha*np.ones((1, N))

    params['W1'] = epsilon*np.random.randn(H, int(N*(N-1)/2))
    params['b1'] = epsilon*np.random.randn(H,1)

    params['W2'] = epsilon*np.random.randn(1, H)
    params['b2'] = epsilon*np.random.randn(1,1)

    params['gamma'] = gamma*np.ones((1,1))

    return params


# def count params, count the number of parameters in the params dictionary
def count_params(params):
    count = 0
    for key in params:
        count += params[key].shape[0]*params[key].shape[1]
    return count










###################### NN AND PSI

@njit
def get_norm(X):
    return np.sqrt(np.sum(X**2, axis=1))


# get distance2
@njit
def get_norm2(X):
    return np.sum(X**2, axis=1)

@njit
def get_relative_distance(X):
    relative_dist = np.zeros(int(X.shape[0]*(X.shape[0]-1)/2))
    for i in range(X.shape[0]):
        for j in range(i):
            relative_dist[i*(i-1)//2+j] = np.sqrt(np.sum((X[i]-X[j])**2))
    return relative_dist


# nn is a function that from X and params g
@njit
def nn(X, params):
    # non interactive part
    W0 = params['W0']

    norm2 = get_norm2(X)

    z0 = np.dot(W0, norm2) 

    # interactive part, use relative distance and tanh
    W1 = params['W1']
    b1 = params['b1']
    relative_dist = get_relative_distance(X)
    relative_dist = relative_dist[:, np.newaxis]
    z1 = np.dot(W1, relative_dist) + b1
    a1 = np.tanh(z1)

    W2 = params['W2']
    b2 = params['b2']
    z2 = np.dot(W2, a1) + b2
    a2 = np.tanh(z2)

    gamma = params['gamma']
    
    return z0 + gamma*a2


@njit
def psi_nn(X, params):
    return np.exp(nn(X, params))



def nn_grad_params(X, params, grad, h=1e-5):
    for key in params:
        if len(params[key].shape) == 1:
            for i in range(params[key].shape[0]):
                params[key][i] += h
                f_plus = nn(X, params)
                params[key][i] -= 2*h
                f_minus = nn(X, params)
                params[key][i] += h
                grad[key][i] = (f_plus - f_minus)/(2*h)
        else:
            for i in range(params[key].shape[0]):
                for j in range(params[key].shape[1]):
                    params[key][i, j] += h
                    f_plus = nn(X, params)
                    params[key][i, j] -= 2*h
                    f_minus = nn(X, params)
                    params[key][i, j] += h
                    grad[key][i, j] = (f_plus - f_minus)/(2*h)

    return grad




















###################### LOCAL ENERGY


def local_kinetic_energy(X, params, h=1e-5):
    grad = np.zeros(X.shape)
    laplacian = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] += h
            f_plus = nn(X, params)
            X[i, j] -= 2*h
            f_minus = nn(X, params)
            X[i, j] += h
            grad[i, j] = (f_plus - f_minus)/(2*h)
            laplacian += (f_plus - 2*nn(X, params) + f_minus)/h**2
    return -0.5*(laplacian + np.sum(grad**2))



@njit
def local_potential_energy(X):
    relative_dist = get_relative_distance(X)
    return 0.5*np.sum(X**2) + np.sum(1/relative_dist)


def local_energies(chain, params):
    energies = np.zeros(chain.shape[0])
    for i in range(chain.shape[0]):
        energies[i] = local_kinetic_energy(chain[i], params) + local_potential_energy(chain[i])
    return energies















###################### METROPOLIS HASTINGS
# def metropolis algorithm,
# input nn (log of wavefunction), params, N, D, H, nsteps, step_size
def metropolis(N, D, 
               params, 
               nsteps, step_size=0.1):
    X = np.random.randn(N, D)
    X_new = np.zeros(X.shape)
    X_chain = np.zeros((nsteps, N, D))
    for i in range(nsteps):
        X_new = X + step_size*np.random.randn(N, D)
        if np.random.rand() < np.exp(2*(nn(X_new, params) - nn(X, params))):
            X = X_new
        X_chain[i] = X
    return X_chain















######## OPTIMIZATION

# calculate the energy radient along parameters
def energy_grad_param(N, D, H,
                      params,
                      X_chain):
    grad_params = init_params_zeros(N, D, H)
    first_term = init_params_zeros(N, D, H)
    second_term = init_params_zeros(N, D, H)
    energy_tot = 0
    for i in range(X_chain.shape[0]):
        local_energy = local_kinetic_energy(X_chain[i].reshape(N,D), params) + local_potential_energy(X_chain[i].reshape(N,D))
        grad_params = nn_grad_params(X_chain[i].reshape(N,D), params, grad_params)
        first_term = sum_params(first_term, scalar_mult_params(grad_params, local_energy))
        second_term = sum_params(second_term, scalar_mult_params(grad_params, 1))
        energy_tot += local_energy 
    
    first_term = scalar_mult_params(first_term, 1/X_chain.shape[0])
    second_term = scalar_mult_params(second_term, energy_tot/(X_chain.shape[0]**2))
    sum_term = sum_params(first_term, scalar_mult_params(second_term, -1))

    return sum_term, energy_tot/X_chain.shape[0]




# def optimization, use batch step to obtain the energy gradient and update the params, do that for optimization step
def optimization(N, D, H, 
                 params, 
                 optimization_steps, batch_size, step_size=0.1,
                 lr=0.01, decay=0.99, verbose=False):
    energies = np.zeros(optimization_steps)
    for i in range(optimization_steps):

        X_chain = metropolis(N, D, params, batch_size, step_size)
        grad_params, energies[i] = energy_grad_param(N, D, H, params, X_chain)
        for key in params:
            params[key] = params[key] - lr*grad_params[key]
        lr = lr*decay
        if verbose:
            print('Step: ', i, 'Energy: ', energies[i])
    return params, energies


# define optimization but with adam
def optimization_adam(N, D, H, 
                      params, 
                      optimization_steps, batch_size, step_size=0.1,
                      lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, verbose=False):
    energies = np.zeros(optimization_steps)
    m = init_params_zeros(N, D, H)
    v = init_params_zeros(N, D, H)
    for i in range(optimization_steps):
        X_chain = metropolis(N, D, params, batch_size, step_size)
        grad_params, energies[i] = energy_grad_param(N, D, H, params, X_chain)
        for key in params:
            m[key] = beta1*m[key] + (1-beta1)*grad_params[key]
            v[key] = beta2*v[key] + (1-beta2)*grad_params[key]**2
            m_hat = m[key]/(1-beta1**(i+1))
            v_hat = v[key]/(1-beta2**(i+1))
            params[key] = params[key] - lr*m_hat/(np.sqrt(v_hat) + epsilon)
        if verbose:
            print('Step: ', i, 'Energy: ', energies[i])
    return params, energies








###################### ERROR ANALYSIS

@njit
def block_transform(energies):
    energies_prime = np.zeros(len(energies)//2)
    for i in range(len(energies)//2):
        energies_prime[i] = 0.5*(energies[2*i] + energies[2*i+1])
    return energies_prime

@njit
def get_block_std(energies):
    energies_prime = energies
    block_std = np.zeros(int(np.log2(len(energies_prime))) + 1)
    block_std[0] = np.std(energies_prime)/sqrt(len(energies_prime) - 1)
    for i in range(len(block_std)-2):
        energies_prime = block_transform(energies_prime)
        block_std[i+1] = np.std(energies_prime)/sqrt(len(energies_prime) - 1)
    return block_std


def get_std_mean_energy(energies, quantiles=0.8):
    block_std = get_block_std(energies)
    return np.sort(block_std)[int(quantiles*len(block_std))]

