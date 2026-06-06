from typing import Sequence

from .module import Parameter
import math

class Optimizer:
    """
    Base class for all optimizers.

    Args:
        parameters: list of Parameter objects to optimize
    """
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters

    def zero_grad(self) -> None:
        """
        Zero out the gradients of all parameters.
        Should be called before each backward pass.
        """
        raise NotImplementedError

    def step(self) -> None:
        """
        Perform a single optimization step.
        Should be called after backward pass to update parameters.
        """
        raise NotImplementedError

# Begin Task 2.4
class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Args:
        parameters: list of Parameter objects to optimize
        lr: learning rate (default 1.0)
    """
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """
        Zero out the gradients of all parameters.

        For each parameter, resets grad to None.
        This should be called before each backward pass to prevent
        gradient accumulation across batches.
        """
        for param in self.parameters:
            if param.value is None:
                continue
            if hasattr(param.value, 'zero_grad_'):
                param.value.zero_grad_()
            elif hasattr(param.value, 'grad'):
                param.value.grad = None

    def step(self) -> None:
        """
        Perform one SGD update step.

        For each parameter: value = value - lr * grad
        """
        for param in self.parameters:
            if param.value is None:
                continue
            if hasattr(param.value, 'grad') and param.value.grad is not None:
                update = param.value.grad * (-self.lr)
                # param.value = param.value + update
                # Use update to keep the Parameter wrapper intact
                new_val = param.value + update
                param.update(new_val)
            elif hasattr(param.value, 'derivative') and param.value.derivative is not None:
                # Scalar fallback
                new_val = param.value - self.lr * param.value.derivative
                param.update(new_val)

    def _print(self) -> None:
        """Debug helper: print parameter shapes and gradients."""
        for param in self.parameters:
            if param.value is None:
                continue
            print(param.value.shape)
            print(param.value.grad)
