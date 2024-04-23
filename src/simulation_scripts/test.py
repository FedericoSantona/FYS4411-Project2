import numpy as np
import jax.numpy as jnp
import jax

N = 2
dim = 3
n_hidden = 10

M = N*dim


a =   np.random.normal(0,0.1,size = M )
b =  np.random.normal(0,0.1,size = n_hidden )
W =  np.random.normal(0,0.1,size = (M , n_hidden) )

r =  np.random.normal(0,0.1,size = (N , dim) )



def wf_closure( r, a, b, W):
        """

        r: (N, dim) array so that r_i is a dim-dimensional vector
        a: (N, dim) array so that a_i is a dim-dimensional vector
        b: (N, 1) array represents the number of hidden nodes
        W: (N_hidden, N , dim) array represents the weights


        OBS: We strongly recommend you work with the wavefunction in log domain.

        """

        r_flat = r.flatten()
        first_sum =  0.25 * jnp.sum((r_flat-a)**2) 

        lntrm =jnp.log( 1+jnp.exp(b+jnp.sum(r_flat[:, None]*W , axis = 0)))

        second_sum = 0.5 * jnp.sum(lntrm)
        
        wf = -first_sum + second_sum
        
        return wf


def grad_wf_closure( r, a , b, W):
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

        exp_term =   1+np.exp(-(b+np.sum(r_flat[:, None]*W , axis = 0)))

        second_term = 0.5 * np.sum(W / exp_term , axis = 1)

        grad = -first_term + second_term

        return grad


def grad_wf_closure_jax( r, a,b,W):
        """
        computes the gradient of the wavefunction with respect to r, but with jax grad
        """

        # Now we use jax.grad to compute the gradient with respect to the first argument (r)
        # Note: jax.grad expects a scalar output, so we sum over the particles to get a single value.
        grad_log_psi = jax.grad(
            lambda positions: jnp.sum(wf_closure(positions, a,b,W)), argnums=0
        )

        return grad_log_psi(r).reshape(-1)

def laplacian_closure( r, a , b , W):
        """
        Analytical expression for the laplacian of the wavefunction
        Here the output is still of shape (M,) conserving the laplacian
        for each visible node
        """
        r_flat = r.flatten()
        
        num = jnp.exp(b+jnp.sum(r_flat[:, None]*W , axis = 0))
        den = (1+jnp.exp(b+jnp.sum(r_flat[:, None]*W , axis = 0)))**2

        term = num/den

        second_term = -0.5 +0.5*jnp.sum(W**2 * term, axis = 1)
        first_term = grad_wf_closure(r, a, b, W)**2

        

        return second_term+first_term

def laplacian_closure_jax( r, a , b , W):
        """
        Computes the Laplacian of the wavefunction for each particle using JAX automatic differentiation.
        r: Position array of shape (n_particles, n_dimensions)
        alpha: Parameter(s) of the wavefunction
        """
        # Compute the Hessian (second derivative matrix) of the wavefunction
        hessian_psi = jax.hessian(
           lambda positions:  wf_closure(positions ,a ,b ,W), argnums=0
        )  # The hessian matrix for our wavefunction
        
        
        
        first_term = jnp.diag(
            hessian_psi(r).reshape(dim*N , dim*N)
        ) # The hessian is nested like a ... onion

        second_term = (grad_wf_closure_jax(r, a , b , W))**2
        

        laplacian = (first_term + second_term)

        

        return laplacian


laplacian_jax = laplacian_closure_jax( r, a,b,W)


laplacian = laplacian_closure( r, a,b,W)


grad = grad_wf_closure( r, a,b,W)
grad_jax = grad_wf_closure_jax( r, a,b,W)

print("laplacian ", laplacian)
print("laplacian_jax ", laplacian_jax)
print("laplacian_diff ",  laplacian_jax - laplacian)

print("grad_diff" , grad_jax - grad)
