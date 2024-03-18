from optimizers.optimizer import Optimizer  # Adjusted import for standalone execution

class Gd(Optimizer):
    """Gradient descent optimizer."""

    def __init__(self, eta):
        """
        Initialize the optimizer.

        Args:
            eta (float): Learning rate.
        """
        super().__init__(eta)

    def step(self, params, grads):
        """
        Update the parameters using gradient descent.

        Args:
            params (list or array): Current parameters.
            grads (list or array): Gradients of the loss with respect to parameters.

        Returns:
            list: Updated parameters.
        """
        updated_params = [p - self.eta * g for p, g in zip(params, grads)]
        #print("updated_params", updated_params)
        return updated_params
