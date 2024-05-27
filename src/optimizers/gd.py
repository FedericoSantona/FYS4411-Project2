from optimizers.optimizer import Optimizer  # Adjusted import for standalone execution
import numpy as np
from simulation_scripts import config

class Gd(Optimizer):
    """Gradient descent optimizer."""

    def __init__(self, eta):
        """
        Initialize the optimizer.

        Args:
            eta (float): Learning rate.
        """
        super().__init__(eta)

    def step(self, params, grads , ite):
        """
        Update the parameters using gradient descent.

        Args:
            params (list or array): Current parameters.
            grads (list or array): Gradients of the loss with respect to parameters.

        Returns:
            list: Updated parameters.
        """

        t =  self.eta# / (1 + ite)
        
        updated_params = [p - t * g for p, g in zip(params, grads)]

        
        return updated_params


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(self, eta, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the optimizer.

        Args:
            eta (float): Learning rate.
            beta1 (float): Decay rate for the first moment estimates.
            beta2 (float): Decay rate for the second moment estimates.
            epsilon (float): Small value to prevent division by zero.
        """
        super().__init__(eta)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0     # Initialization of the time step (for bias correction)

    def step(self, params, grads , ite):
        """
        Update the parameters using the Adam optimization algorithm.

        Args:
            params (list or array): Current parameters.
            grads (list or array): Gradients of the loss with respect to parameters.

        Returns:
            list: Updated parameters.
        """
        
        self.m = [0] * len(params)
        self.v = [0] * len(params)

        self.t += 1

        t_ =  self.eta# / (1 + 5 * (ite/ config.nparticles) )

        updated_params = []
        for p, g, m, v in zip(params, grads, self.m, self.v):

           
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g ** 2)

            # Compute bias-corrected first and second moment estimates
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

           # breakpoint()
            # Update parameters
            p -= t_ * m_hat / (v_hat ** 0.5 + self.epsilon)

            updated_params.append(p)

        # Update the moment vectors
        self.m = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(self.m, grads)]
        self.v = [self.beta2 * v + (1 - self.beta2) * (g ** 2) for v, g in zip(self.v, grads)]
        
        return updated_params

