import numpy as np

x = np.array([-0.85 , -0.73 , 0.48 , 1.01 , 0.5 , -0.93 , -1.58 , 1.26 , 1.19 , -0.04])


variance = np.sum((x - np.mean(x))**2) / ( len(x) -1)

print("variance " , variance)

std = np.sqrt(variance) / np.sqrt(len(x))


print("standard deviation" , std)


mean = np.mean(x)

print(" mean " , mean)