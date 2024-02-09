import jax.numpy as jnp

# Define some sample data
accept = jnp.array([True, False, True])
proposed_positions = jnp.array([[1, 2], [3, 4], [5, 6]])
initial_positions = jnp.array([[10, 20], [30, 40], [50, 60]])

# Apply jnp.where to the sample data
new_positions = jnp.where(accept[:, None], proposed_positions, initial_positions)

# Print the result
print("New positions:")
print(new_positions)