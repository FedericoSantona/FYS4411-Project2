import jax.numpy as np
from jax import grad, jit,pmap


def f(x):
    return float(3)

gradf = grad(f)


for i in range(2):
    p = float(i)
    print(i,gradf(p))
