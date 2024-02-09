# DISCLAIMER: Idea and code structure from blackjax
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Union

import jax.numpy as jnp
import numpy as np

# from typing import Callable

Array = Union[np.ndarray, jnp.ndarray]  # either numpy or jax numpy array
PyTree = Union[
    Array, Iterable[Array], Mapping[Any, Array]
]  # either array, iterable of arrays or mapping of arrays

from dataclasses import dataclass


@dataclass(frozen=False)
class State:
    positions: PyTree
    logp: Union[float, PyTree]
    n_accepted: int
    delta: int

    def __init__(self, positions, logp, n_accepted=0, delta=0):
        self.positions = positions
        self.logp = logp
        self.n_accepted = n_accepted
        self.delta = delta

    def create_batch_of_states(self, batch_size):
        """
        # TODO: check if batch states are immutable because of the jnp
        """
        # Replicate each property of the state
        batch_positions = jnp.array([self.positions] * batch_size)
        batch_logp = jnp.array([self.logp] * batch_size)
        batch_n_accepted = jnp.array([self.n_accepted] * batch_size)
        batch_delta = jnp.array([self.delta] * batch_size)

        # Create a new State object with these batched properties
        batch_state = State(batch_positions, batch_logp, batch_n_accepted, batch_delta)
        return batch_state
