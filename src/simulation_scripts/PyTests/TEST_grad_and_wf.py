import jax
import jax.numpy as jnp
import numpy as np



print("\n--------------------------------")
print("This file tests the implimentation of the gradient of the wavefunction w.r.t the variational parameters,")
print("both the Jax and numpy implimentationas are compared, also Morten's implimentation is compared to our numpy implimentation.\n")
print("The last test checks if our actual implimented WF matches with Morten's for a given tolerance.")
print("--------------------------------\n\n\n")

tol = 10E-6
tol2 = 10E-8
Np = 2
d = 2
nhidden = 10
M =Np*d
nbatch = 1
r = np.random.normal(0,1,size = (nbatch,Np , d))
a =  np.random.normal(0,0.1,size = M )
b =  np.random.normal(0,0.1,size = nhidden )
W =  np.random.normal(0,0.1,size = (M , nhidden) )

# Here we test the diff between the gradient of the wavefunction w.r.t the variational
#parameters, our numerical and analytical implimentations are compared to Morten's analytical for the 
#2 particle, 2 dim case.


def Qfac(r,b,w):
        Q = np.zeros((nhidden), np.double)
        temp = np.zeros((nhidden), np.double)
        for ih in range(nhidden):
                temp[ih] = (r*w[:,:,ih]).sum()
        Q = b + temp
        return Q

#Morten's derivative, changed with a factor of 2 as to match our expressions
def DerivativeWFansatz(r,a,b,w):
        sigma=1.0
        sig2 = sigma**2
        Q = Qfac(r,b,w)
        WfDer = np.empty((3,),dtype=object)
        WfDer = [np.copy(a),np.copy(b),np.copy(w)]
        WfDer[0] = (r-a)/(2*sig2)
        WfDer[1] = 1 / (2*(1 + np.exp(-Q)))
        for ih in range(nhidden):
                WfDer[2][:,:,ih] = r[:,:] / (2*sig2*(1+np.exp(-Q[ih])))

        return WfDer[0], WfDer[1], WfDer[2]


Mgrad_a,Mgrad_b,Mgrad_W = DerivativeWFansatz(r,a.reshape(Np,d),b,W.reshape(Np,d,nhidden))

def grads_closure(r, a, b, W):
        """
        Computes the gradient of the wavefunction with respect to the variational parameters analytically

        Here r comes in shape (n_batch , n_particles , n_dim)
        """

        r_flat = r.reshape(nbatch , -1)


        #grad_a is of shape (N_batch , M)
        grad_a = (0.5 * (r_flat - a))


        # grad_b is of shape (n_batch , n_hidden)
        grad_b = 1 / (2*( 1+jnp.exp(-(b+jnp.sum(r_flat[:,: ,None]*W[None,:,:] , axis = 1)))))

        #grad_W is of shape (n_batch ,  M * N_hidden )
        
        grad_W  = (r_flat[:,:,None] * grad_b[:,None,:]).reshape(nbatch , M * nhidden)
        

        
        return grad_a , grad_b , grad_W
Ngrad_a, Ngrad_b, Ngrad_W = grads_closure(r,a,b,W)






def wf_closure(r, a, b, W):
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


#Here we have the jax implimentation
grad_a = jax.vmap(jax.grad(wf_closure,1),(0,None,None,None),0)(r,a,b,W)
grad_b = jax.vmap(jax.grad(wf_closure,2),(0,None,None,None),0)(r,a,b,W)
grad_W = jax.vmap(jax.grad(wf_closure,3),(0,None,None,None),0)(r,a,b,W).reshape(nbatch,M*nhidden)


# First we test for a
grad_diff_a = grad_a-Ngrad_a
print("----- TEST 1: Gradient of WF w.r.t a -----")
if np.abs(np.max(grad_diff_a)) > tol or np.sum(Ngrad_a-Mgrad_a.reshape(nbatch,d*Np))>tol:
        raise ValueError(" The numerical gradient of the WF w.r.t a is not equal the analytical!\n")
else: 
        print("Jax-Numpy equivalence: Pass\nNumpy-Morten equivalence: Pass\n") 

print("The largest difference between numerical and analytical gradient of Wf w.r.t a is ", np.max(grad_a-Ngrad_a))

#Then we test for b
grad_diff_b = grad_b-Ngrad_b

print("\n")
print("----- TEST 2: Gradient of WF w.r.t b -----")


if np.abs(np.max(grad_diff_b)) > tol or np.abs(Mgrad_b.sum()-Ngrad_b.sum())>tol: 
        raise ValueError(" The numerical gradient of the WF w.r.t b is not equal the analytical!\n")
else: 
        print("Jax-Numpy equivalence: Pass\nNumpy-Morten equivalence: Pass\n") 

print("The largest difference between numerical and analytical gradient of Wf w.r.t b is ", np.max(grad_b-Ngrad_b))

print("\n")

#Then we test for W
grad_diff_W = grad_W-Ngrad_W
print("----- TEST 3: Gradient of WF w.r.t W -----")
if np.abs(np.max(grad_diff_W)) > tol or np.abs( Mgrad_W.sum()-Ngrad_W.sum())>tol:
        raise ValueError(" The numerical gradient of the WF w.r.t W is not equal the analytical!\n")

else: 
        print("Jax-Numpy equivalence: Pass\nNumpy-Morten equivalence: Pass\n") 

print("The largest difference between numerical and analytical gradient of Wf w.r.t W is ", np.max(grad_W-Ngrad_W))


print("\n")
def wf_closure(r, a, b, W):
        """

        r: (N, dim) array so that r_i is a dim-dimensional vector
        a: (N, dim) array so that a_i is a dim-dimensional vector
        b: (N, 1) array represents the number of hidden nodes
        W: (N_hidden, N , dim) array represents the weights


        OBS: We strongly recommend you work with the wavefunction in log domain.

        """

        r_flat = r.flatten()
        first_sum =  0.25 * np.sum((r_flat-a)**2) 

        lntrm =np.log( 1+np.exp(b+np.sum(r_flat[:, None]*W , axis = 0)))

        second_sum = 0.5 * np.sum(lntrm)
        
        wf = -first_sum + second_sum
        
        return wf

def WaveFunction(r,a,b,w):
        sigma=1.0
        sig2 = sigma**2
        Psi1 = 0.0
        Psi2 = 1.0
        Q = Qfac(r,b,w)
        for iq in range(Np):
                for ix in range(d):
                        Psi1 += (r[iq,ix]-a[iq,ix])**2
        for ih in range(nhidden):
                Psi2 *= (1.0 + np.exp(Q[ih]))
        Psi1 = np.exp(-Psi1/(2*sig2))
        return Psi1*Psi2



r = np.random.normal(0,0.1,size = (Np , d))
a =  np.random.normal(0,0.1,size = M )
b =  np.random.normal(0,0.1,size = nhidden )
W =  np.random.normal(0,0.1,size = (M , nhidden) )

# We have a factor 2 diff as we define the WF differently
Our_WF = 2*wf_closure(r,a,b,W)
Morten_WF = np.log(WaveFunction(r,a.reshape(Np,d),b,W.reshape(Np,d,nhidden)))
wfdiff = np.abs(Our_WF-Morten_WF)
print("\n----- WF Test -----")
if wfdiff > tol:
        raise ValueError("Wf is wrong\n")
else: 
        print("Implimented WF - Morten's WF equivalence test: Pass\n")

print("Wfdiff ", np.log(WaveFunction(r,a.reshape(Np,d),b,W.reshape(Np,d,nhidden)))-2*wf_closure(r,a,b,W) )


# Next we test gradients of wavefunction w.r.t 