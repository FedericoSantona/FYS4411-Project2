import numpy as np
import matplotlib.pyplot as plt

#Optimal parameter for two particles w int in HO in 3d
alpha = 0.25
#Optimal alpha for non int
alpha_nonInt = 0.5

#The a we were asked to use
a = 0.0043


umax = 100
N = 10000
du = (umax-a)/(N-1)

tol = 2

rmax = np.sqrt(tol*np.log(10)/alpha)
r1 = np.linspace(1e-7,rmax,N)

uvec = np.linspace(a,umax,N)

rho = np.zeros(N)


index = 0
#Forward Euler to solve the integral per r1-coord
for r1i in r1:
    
    #Calculating each part of the sum
    term1 = du* uvec
    term2 = np.sinh(4*alpha*uvec*r1i)
    term3 = np.exp(-2*alpha*uvec**2)
    term4 = (1-a/uvec)**2
    sumsum = np.sum(term1*term2*term3*term4)
    
    rho_i = np.pi/(alpha*r1i)*np.exp(-4*alpha*r1i**2)*sumsum
    
    rho[index] = rho_i
    
    index +=1

#Normalising
rho = rho/np.trapz(rho,r1)    
#The analytical expression    
rho_noInt = (np.pi/(8*alpha_nonInt))**(-1/2) * np.exp(-2*alpha_nonInt*r1**2)

normalisation_tolerance = 10e-3

#Test to see if analytical and numerical density is normalised
if abs((np.trapz(rho_noInt,r1) - 1)) > normalisation_tolerance:
    raise AssertionError(" Analytical density is not normalised!")
elif abs((np.trapz(rho,r1) - 1)) > normalisation_tolerance:
    raise AssertionError(" Numerical twobody density is not normalised!")

plt.plot(r1,rho,label = "Interactions")

plt.plot(r1,rho_noInt,"-",label ="No interactions")
plt.legend()
plt.xlabel("$r_1 [-]$")
plt.ylabel(r"$\rho(r_1) [-]$")
plt.savefig("onebody_comp.pdf")
plt.show()

