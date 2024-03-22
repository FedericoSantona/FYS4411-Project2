import jax.numpy as jnp
from jax import vmap

# Define a simple function that adds two numbers
def add(x, y):
    return x + y

# Vectorize the function using vmap
vectorized_add = vmap(add)

# Prepare input arrays
x_array = jnp.array([1, 2, 3, 4])
y_array = jnp.array([5, 6, 7, 8])

# Execute the vectorized function on arrays
result = vectorized_add(x_array, y_array)

print(result)
