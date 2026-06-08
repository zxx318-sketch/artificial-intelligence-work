from typing import Sequence, Dict, Any

from .module import Parameter
import numpy as np

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
        weight_decay: L2 regularization strength (default 0.0 means no L2)
                      The update becomes: w = w - lr * (grad + weight_decay * w)
    """
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0, weight_decay: float = 0.0):
        super().__init__(parameters)
        self.lr = lr
        self.weight_decay = weight_decay

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

        For each parameter: value = value - lr * (grad + weight_decay * value)
        When weight_decay=0, this is equivalent to vanilla SGD (no L2).
        """
        for param in self.parameters:
            if param.value is None:
                continue
            if hasattr(param.value, 'grad') and param.value.grad is not None:
                # grad_with_l2 = grad + weight_decay * w
                grad = param.value.grad
                if self.weight_decay > 0.0:
                    grad = grad + param.value * self.weight_decay
                update = grad * (-self.lr)
                new_val = param.value + update
                param.update(new_val)
            elif hasattr(param.value, 'derivative') and param.value.derivative is not None:
                # Scalar fallback
                w = param.value
                d = param.value.derivative
                if self.weight_decay > 0.0:
                    d = d + w * self.weight_decay
                new_val = w - self.lr * d
                param.update(new_val)

    def _print(self) -> None:
        """Debug helper: print parameter shapes and gradients."""
        for param in self.parameters:
            if param.value is None:
                continue
            print(param.value.shape)
            print(param.value.grad)


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Combines the advantages of RMSProp and momentum:
      m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
      m_hat = m_t / (1 - beta1^t)
      v_hat = v_t / (1 - beta2^t)
      w = w - lr * m_hat / (sqrt(v_hat) + eps)

    Args:
        parameters: list of Parameter objects to optimize
        lr: learning rate (default 1e-3)
        betas: coefficients for computing running averages of gradient and its square (default (0.9, 0.999))
        eps: term added to denominator for numerical stability (default 1e-8)
        weight_decay: L2 regularization strength (default 0.0)
    """
    def __init__(
        self,
        parameters: Sequence[Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # timestep counter

        # Per-parameter state: dict mapping param id -> (m_np, v_np) as numpy arrays
        # Using numpy avoids building computation graphs for optimizer states
        self._state: Dict[int, Any] = {}

    def zero_grad(self) -> None:
        """Zero out the gradients of all parameters."""
        for param in self.parameters:
            if param.value is None:
                continue
            if hasattr(param.value, 'zero_grad_'):
                param.value.zero_grad_()
            elif hasattr(param.value, 'grad'):
                param.value.grad = None

    def step(self) -> None:
        """Perform one Adam update step."""
        self.t += 1

        for param in self.parameters:
            if param.value is None:
                continue

            # Only handle Tensor parameters with grad
            if not (hasattr(param.value, 'grad') and param.value.grad is not None):
                continue

            grad = param.value.grad
            grad_np = grad.to_numpy()
            w_np = param.value.to_numpy()
            pid = id(param)

            # Initialize state for this parameter if not exists
            if pid not in self._state:
                self._state[pid] = {
                    'm': np.zeros_like(w_np),
                    'v': np.zeros_like(w_np),
                }

            m = self._state[pid]['m']
            v = self._state[pid]['v']

            # weight_decay (decoupled, applied directly to weights)
            if self.weight_decay > 0.0:
                grad_np = grad_np + self.weight_decay * w_np

            # Update biased first and second moment estimates
            m = self.beta1 * m + (1.0 - self.beta1) * grad_np
            v = self.beta2 * v + (1.0 - self.beta2) * (grad_np ** 2)

            # Bias correction
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)

            # Update weights
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            new_w_np = w_np - update

            # Store updated state
            self._state[pid] = {'m': m, 'v': v}

            # Update parameter with new value (convert back to tensor)
            from .tensor_functions import tensor_from_numpy
            new_val = tensor_from_numpy(new_w_np.astype(np.float32), backend=param.value.backend)
            param.update(new_val)

    def _print(self) -> None:
        """Debug helper: print parameter shapes and gradients."""
        for param in self.parameters:
            if param.value is None:
                continue
            print(param.value.shape)
            print(param.value.grad)
