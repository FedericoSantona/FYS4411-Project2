import jax
import jax.numpy as jnp
import numpy as np

Np = 2
d = 2
nhidden = 10
M =Np*d
nbatch = 200
r = np.random.normal(0,0.1,size = (Np , d))
a =  np.random.normal(0,0.1,size = M )
b =  np.random.normal(0,0.1,size = nhidden )
W =  np.random.normal(0,0.1,size = (M , nhidden) )


def Qfac(r,b,w):
    Q = np.zeros((nhidden), np.double)
    temp = np.zeros((nhidden), np.double)

    for ih in range(nhidden):
        temp[ih] = (r*w[:,:,ih]).sum()

    Q = b + temp
    return Q

# NB we changd Morten's code as he defined WF differently
def LocalEnergy(r,a,b,w):
    sigma=1.0
    sig2 = sigma**2
    locenergy = 0.0
    Q = Qfac(r,b,w)
    for iq in range(Np):
        for ix in range(d):
            sum1 = 0.0
            sum2 = 0.0
            for ih in range(nhidden):
                sum1 += w[iq,ix,ih]/(1+np.exp(-Q[ih]))
                sum2 += w[iq,ix,ih]**2 * np.exp(Q[ih]) / (1.0 + np.exp(Q[ih]))**2
            dlnpsi1 = -(r[iq,ix] - a[iq,ix]) /sig2 + sum1/sig2
            dlnpsi2 = -1/(sig2) + sum2/sig2**2
            locenergy += 0.5*(-dlnpsi1*dlnpsi1/4 - dlnpsi2/2 + r[iq,ix]**2)
    return locenergy



def grad_wf_closure(r, a , b, W):
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


def laplacian_closure( r, a , b , W):
        """
        Analytical expression for the laplacian of the wavefunction
        Here the output is still of shape (M,) conserving the laplacian
        for each visible node
        """
        r_flat = r.flatten()
        
        num = np.exp(b+np.sum(r_flat[:, None]*W , axis = 0))
        den = (1+num)**2

        term = num/den

        laplacian = -0.5 +0.5*np.sum(W**2 * term, axis = 1)

        return laplacian


def non_int_energy(r):

        first_term = grad_wf_closure(r,a,b,W)**2
        second_term = laplacian_closure(r,a,b,W)
        third_term = 1 *  (r**2).flatten()

        #The sum without specific axis is the sum of all elements in the array i.e. returns a scalar
        non_int_energy =  0.5*np.sum(-first_term - second_term + third_term) 

        return non_int_energy


tol = 10E-6
Our_E = non_int_energy(r)
Mort_E = LocalEnergy(r,a.reshape(Np,d),b,W.reshape(Np,d,nhidden))
print("----- E_loc test -----\n")
if np.abs((Our_E-Mort_E))>tol:
    raise ValueError(" Local energy is wrong")
else: 
     print("Local energy is correct.\n")
     print(f"The difference between our and Morten's E_loc is within tolerance of {tol} \n")
print("Diff in E_loc is ", np.abs((Our_E-Mort_E)))