import numpy as np
from matplotlib import pyplot as plt

N_particle = 4
n_dim = 3
N_hidden = 2

r = np.random.rand(N_particle, n_dim)


a =  np.random.normal(0,1,size = (N_particle, n_dim) )
b =  np.random.normal(0,1,size = N_hidden )
W =  np.random.normal(0,1,size = (N_hidden, N_particle,  n_dim) )

#b = np.reshape(b, (N_hidden,1))

r_shape = r.shape
a_shape = a.shape
b_shape = b.shape
W_shape = W.shape
     
#Wafe function
first_sum =  np.sum((r-a)**2, axis = 1) /4

apoo = b+np.sum(np.sum(r[None,:,:]*W,axis = 2), axis = 1)


lntrm = 1+np.exp(b+np.sum(np.sum(r[None,:,:]*W,axis = 2), axis = 1))
second_sum = np.sum(lntrm )/2
wf = -first_sum + second_sum

#gradient

first_term = 0.5 * (r - a) 

exp_term = 1+np.exp(-(b+np.sum(np.sum(r[None,:,:]*W,axis = 2), axis = 1)))
second_term = 0.5 * np.sum(W / exp_term [:,None,None], axis = 0)


#laplacian 

num = np.exp(b+np.sum(np.sum(r[None,:,:]*W, axis = 2), axis = 1))
den = (1+np.exp(b+np.sum(np.sum(r[None,:,:]*W,axis = 2), axis = 1)))**2

term = num/den

laplacian1 = -0.5 +0.5*np.sum(W**2 * term[:,None,None], axis = 0)

laplacian2 = -0.5 +0.5*np.sum(W**2 * term[:,None,None], axis = (0,2))



#grad for params


grad_a = 0.5 * (r - a)
        
grad_b = 1 / (2*( 1+np.exp(-(b+np.sum(np.sum(r[None,:,:]*W,axis = 2), axis = 1)))))

grad_W  = r * grad_b[:,None,None]



array = np.array([[1,2,3], [4,5,6], [7,8,9]])


print(array)

print("array flattened" , array.flatten())
 



