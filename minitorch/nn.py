from typing import Tuple, Optional
import math

from . import operators
from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor, tensor_from_numpy
from .module import Module, Parameter
from .tensor_ops import SimpleBackend
import numpy as np

# ==============================================================================
# Part 1: Basic functions
# ==============================================================================

def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_width = width // kw
    new_height = height // kh

    x = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    x = x.view(batch, channel, new_height, new_width, kh * kw)
    return x, new_height, new_width


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        out = input.f.max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        input, out = ctx.saved_values
        return (out == input) * grad_output, 0.0

        
def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))

# Begin Task 2.3

# ==============================================================================
# Part 2: Modules (Neural Network Layers)
# ==============================================================================

class Linear(Module):
    """
    A fully connected linear layer (y = x @ W + b).

    Args:
        in_size: size of each input sample
        out_size: size of each output sample
        bias: whether to include a bias term (default True)
        init_mode: weight initialization method, "he" for ReLU, "xavier" for Sigmoid/Tanh
    """
    def __init__(self, in_size: int, out_size: int, bias: bool = True, init_mode: str = "he"):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        # ---- 权重初始化 ----
        # He 初始化: std = sqrt(2 / in_size)    (适合 ReLU)
        # Xavier 初始化: std = sqrt(2 / (in_size + out_size)) (适合 Sigmoid/Tanh)
        if init_mode == "he":
            std = math.sqrt(2.0 / in_size)
        elif init_mode == "xavier":
            std = math.sqrt(2.0 / (in_size + out_size))
        else:
            std = math.sqrt(2.0 / in_size)

        # 用标准正态分布 N(0, std^2) 初始化权重
        weight_data = np.random.randn(in_size, out_size).astype(np.float32) * std
        bias_data = np.zeros(out_size, dtype=np.float32)

        # 直接将 Parameter 赋给 self.weight / self.bias
        # __setattr__ 会自动将它们注册到 _parameters 中
        self.weight = Parameter(
            tensor_from_numpy(weight_data, backend=SimpleBackend),
            name="weight"
        )
        if bias:
            self.bias = Parameter(
                tensor_from_numpy(bias_data, backend=SimpleBackend),
                name="bias"
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        y = x @ W + b
        x  shape: (batch, in_size)
        W  shape: (in_size, out_size)
        y  shape: (batch, out_size)
        """
        batch, in_size = x.shape
        # 确保输入是2D (batch, in_size)
        x_flat = x.contiguous().view(batch, in_size)

        # 矩阵乘法: (batch, in_size) @ (in_size, out_size) -> (batch, out_size)
        out = x_flat @ self.weight.value

        # 添加偏置 (广播)
        if self.bias is not None:
            out = out + self.bias.value

        return out

class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        # mask ~ Bernoulli(1-p), scaled by 1/(1-p)
        mask = rand(x.shape, backend=x.backend)
        mask = mask > self.p  # 需要 LT 支持（已有）
        # 缩放：保留的神经元除以 (1-p)
        scale = 1.0 / (1.0 - self.p)
        return x * mask * scale



class Conv2d(Module):
    """
    A 2D convolutional layer.

    Uses the direct convolution implementation from tensor_functions.Conv2d.

    Args:
        in_channels: number of channels in the input image
        out_channels: number of channels produced by the convolution
        kh: kernel height
        kw: kernel width
        stride: convolution stride (default 1)
        padding: zero-padding (default 0)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kh: int,
        kw: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh = kh
        self.kw = kw
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        # Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
        fan_in = in_channels * kh * kw
        fan_out = out_channels * kh * kw
        std = math.sqrt(2.0 / (fan_in + fan_out))

        # weight: (out_channels, in_channels, kh, kw)
        weight_data = np.random.randn(out_channels, in_channels, kh, kw).astype(np.float32) * std
        # bias: (out_channels,)
        bias_data = np.zeros(out_channels, dtype=np.float32)

        self.weight = Parameter(
            tensor_from_numpy(weight_data, backend=SimpleBackend),
            name="weight"
        )
        self.bias = Parameter(
            tensor_from_numpy(bias_data, backend=SimpleBackend),
            name="bias"
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: conv2d(x, weight) + bias

        Args:
            x: (N, C_in, H, W) input tensor

        Returns:
            (N, C_out, H_out, W_out) output tensor
        """
        from .tensor_functions import Conv2d as Conv2dFn
        from .tensor_functions import tensor as make_tensor
        
        # Wrap stride/padding as tensors so Function.apply can handle them
        stride_t = make_tensor(list(self.stride))
        padding_t = make_tensor(list(self.padding))
        return Conv2dFn.apply(x, self.weight.value, self.bias.value,
                              stride_t, padding_t)


class MaxPool2d(Module):
    """
    A 2D max pooling layer.

    Args:
        kernel: (height, width) of the pooling kernel
        stride: stride of the pooling (defaults to kernel size if None)
    """
    def __init__(self, kernel: Tuple[int, int], stride: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.kernel = kernel
        self.stride = stride if stride is not None else kernel

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for max pooling.

        Steps:
          1. Use im2col to extract patches from the input
          2. Take max over the patch dimension
          3. Reshape back to 4D output

        Args:
            x: (N, C, H, W) input tensor

        Returns:
            (N, C, H_out, W_out) output tensor
        """
        kh, kw = self.kernel
        sh, sw = self.stride

        # Use tile to reshape into patches: (N, C, H_out, W_out, kh*kw)
        tiled, new_h, new_w = tile(x, self.kernel)

        # Max over last dim (the patch dim)
        pooled = max(tiled, 4)  # dim=4 is the last dim with size kh*kw

        # Reshape to (N, C, H_out, W_out)
        batch, channel = x.shape[0], x.shape[1]
        out = pooled.contiguous().view(batch, channel, new_h, new_w)

        return out


class Flatten(Module):
    """
    Flattens a multidimensional tensor into a 2D matrix.
    Assumes the first dimension is the batch size (N).
    e.g., (N, C, H, W) -> (N, C * H * W)
    """
    def forward(self, x: Tensor) -> Tensor:

        # 1. Batch size
        batch = x.shape[0]
        # 2. Flatten remaining dimensions
        feat_dim = int(np.prod(x.shape[1:]))
        # 3. Reshape
        return x.contiguous().view(batch, feat_dim)


class Sequential(Module):
    """
    A sequential container of modules.
    Modules will be added to it in the order they are passed in the constructor.
    The forward pass of Sequential passes the input through each module in order.
    """
    def __init__(self, *modules: Module):
        super().__init__()
        self.modules_list = list(modules)
        for i, module in enumerate(self.modules_list):
            # Register submodule under a numbered key
            self.__setattr__(f"_{i}", module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules_list:
            x = module(x)
        return x
        


# ==============================================================================
# Part 3: Activation Function
# ==============================================================================

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    
class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

# ==============================================================================
# Part 4: Loss Function
# ==============================================================================

class CrossEntropyLossFunction(Function):
    """Fused cross-entropy loss: forward/backward in one node."""

    @staticmethod
    def forward(ctx, predictions, targets):
        logits_np = predictions.to_numpy()
        labels_np = targets.to_numpy()
        batch = logits_np.shape[0]

        # Numerically stable log-softmax
        max_logits = np.max(logits_np, axis=1, keepdims=True)
        shifted = logits_np - max_logits
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        log_probs = shifted - np.log(sum_exp)

        # Cross-entropy: -mean(sum(targets * log_probs, axis=1))
        nll = -np.sum(labels_np * log_probs, axis=1)
        loss_val = np.mean(nll)

        # Save for backward: softmax(probs) and labels
        softmax = exp_shifted / sum_exp
        ctx.save_for_backward(softmax, labels_np, batch)

        return tensor_from_numpy(
            np.array([loss_val], dtype=np.float32),
            backend=predictions.backend,
        )

    @staticmethod
    def backward(ctx, grad_output):
        softmax, labels_np, batch = ctx.saved_values
        grad_scalar = grad_output.to_numpy().item()
        grad_np = grad_scalar * (softmax - labels_np) / batch
        grad_tensor = tensor_from_numpy(
            grad_np.astype(np.float32),
            backend=grad_output.backend,
        )
        return grad_tensor, None


class CrossEntropyLoss(Module):
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Cross-entropy loss with log-softmax for numerical stability.

        Args:
            predictions: (batch, num_classes) raw logits (before softmax)
            targets: (batch, num_classes) one-hot encoded labels

        Returns:
            scalar loss = -mean( sum(targets * log_softmax(predictions), dim=1) )
        """
        return CrossEntropyLossFunction.apply(predictions, targets)

# End Task 2.3

