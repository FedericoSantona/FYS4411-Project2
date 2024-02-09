class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, eta):
        """
        Initialize the optimizer.

        Args:
            eta (float): Learning rate.
        """
        self.eta = eta

    def step(self, params, grads):
        """
        Update the parameters.

        Args:
            params (list or array): Current parameters.
            grads (list or array): Gradients of the loss with respect to parameters.

        Returns:
            Updated parameters.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
