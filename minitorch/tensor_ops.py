from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    MAX_DIMS,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
    IndexingError,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        pass

    @staticmethod
    def im2col(a: Tensor, kernel_shape: Tuple[int, int], stride: Tuple[int, int]) -> Tensor:
        pass
    
    @staticmethod
    def col2im(a: Tensor, output_shape: Tuple[int, ...], kernel_shape: Tuple[int, int], stride: Tuple[int, int]) -> Tensor:
        pass

    @staticmethod
    def conv2d(
        a: Tensor, 
        b: Tensor, 
        stride: Tuple[int, int], 
        padding: Tuple[int, int]
    ) -> Tensor:
        """
        Direct convolution implementation without im2col.
        Args:
            a: Input tensor (N, C_in, H_in, W_in)
            b: Weight tensor (C_out, C_in, KH, KW)
            stride: (stride_h, stride_w)
            padding: (pad_h, pad_w)
        Returns:
            Output tensor (N, C_out, H_out, W_out)
        """
        pass

    @staticmethod
    def conv2d_backward(
        a: Tensor,
        b: Tensor,
        grad_out: Tensor,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        """
        Direct convolution backward (no im2col).

        Given forward:
          out[n, c_out, h_out, w_out] =
            sum_{c_in, kh, kw} a[n, c_in, h_out*sh+kh-ph, w_out*sw+kw-pw] * b[c_out, c_in, kh, kw]

        Computes:
          d_a (same shape as a), d_b (same shape as b)

        Args:
            a: Input tensor (N, C_in, H_in, W_in)
            b: Weight tensor (C_out, C_in, KH, KW)
            grad_out: d_out tensor (N, C_out, H_out, W_out)
            stride: (stride_h, stride_w)
            padding: (pad_h, pad_w)

        Returns:
            (d_a, d_b)
        """
        pass
    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)
        self.tanh_map = ops.map(operators.tanh)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)
        self.pow_scalar_zip = ops.zip(operators.pow)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.max_reduce = ops.reduce(operators.max, -1e9)
        self.matrix_multiply = ops.matrix_multiply
        self.conv2d = ops.conv2d
        self.conv2d_backward = ops.conv2d_backward

        # Extra credit operations
        self.im2col = ops.im2col
        self.col2im = ops.col2im

        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
            new tensor data
        """

        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(out._tensor, a._tensor)
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
            :class:`TensorData` : new tensor data
        """

        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(out._tensor, a._tensor, b._tensor)
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """
        Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
            :class:`TensorData` : new tensor
        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(out._tensor, a._tensor, dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """
        Matrix multiplication of two tensors.
        
        Args:
            a: First tensor with shape (..., m, n)
            b: Second tensor with shape (..., n, p)
            
        Returns:
            Tensor with shape (..., m, p)
        """
        # Get the shapes
        a_shape = a.shape
        b_shape = b.shape
        
        # Check if we can perform matrix multiplication
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError("Both tensors must be at least 2D for matrix multiplication")
        
        # For matrix multiplication, the last two dimensions must be compatible
        # a: (..., m, n), b: (..., n, p) -> result: (..., m, p)
        m, n = a_shape[-2], a_shape[-1]
        n2, p = b_shape[-2], b_shape[-1]
        
        if n != n2:
            raise ValueError(f"Matrix multiplication requires compatible dimensions: {a_shape} and {b_shape}")
        
        # Handle broadcasting for higher dimensions
        # Broadcast the leading dimensions
        if len(a_shape) > 2 or len(b_shape) > 2:
            # Get the leading dimensions (excluding the last 2)
            a_leading = a_shape[:-2]
            b_leading = b_shape[:-2]
            
            # Broadcast the leading dimensions
            try:
                leading_shape = shape_broadcast(a_leading, b_leading)
            except IndexingError:
                raise ValueError(f"Cannot broadcast leading dimensions: {a_leading} and {b_leading}")
            
            # Create output shape
            out_shape = leading_shape + (m, p)
        else:
            out_shape = (m, p)
        
        # Create output tensor
        out = a.zeros(out_shape)
        
        # Get the tensor data
        a_storage, a_shape_array, a_strides = a.tuple()
        b_storage, b_shape_array, b_strides = b.tuple()
        out_storage, out_shape_array, out_strides = out.tuple()
        
        # Perform matrix multiplication
        # For each output position (i, j), compute sum over k
        for i in range(m):
            for j in range(p):
                # Compute C[i,j] = sum(A[i,k] * B[k,j] for k in range(n))
                sum_val = 0.0
                for k in range(n):
                    # Get A[i,k]
                    a_index = np.array([i, k])
                    a_pos = index_to_position(a_index, a_strides)
                    a_val = a_storage[a_pos]
                    
                    # Get B[k,j]
                    b_index = np.array([k, j])
                    b_pos = index_to_position(b_index, b_strides)
                    b_val = b_storage[b_pos]
                    
                    # Accumulate the product
                    sum_val += a_val * b_val
                
                # Set the result
                out_index = np.array([i, j])
                out_pos = index_to_position(out_index, out_strides)
                out_storage[out_pos] = sum_val
        
        return out

    from numpy.lib.stride_tricks import as_strided

    @staticmethod
    def im2col(a: "Tensor", kernel_shape: Tuple[int, int], stride: Tuple[int, int]) -> "Tensor":
        """
        im2col: 对输入的4D张量 a (N, C, H, W)，抽取 patch 展平为 2D 张量 [C * Kh * Kw, N * Oh * Ow]
        """
        # 不使用
        raise NotImplementedError("im2col is not implemented")
    
    @staticmethod
    def col2im(a: "Tensor", output_shape: Tuple[int, ...], kernel_shape: Tuple[int, int], stride: Tuple[int, int]) -> "Tensor":
        """
        本质：im2col 的逆过程，涉及累加重叠点。
        """
        # 不使用
        raise NotImplementedError("col2im is not implemented")

    @staticmethod
    def conv2d(
        a: "Tensor", 
        b: "Tensor", 
        stride: Tuple[int, int], 
        padding: Tuple[int, int]
    ) -> "Tensor":
        """
        直接卷积实现 (Direct Conv2d)，不使用 im2col。
        
        Args:
            a: Input tensor (N, C_in, H_in, W_in)
            b: Weight tensor (C_out, C_in, KH, KW)
            stride: (stride_h, stride_w)
            padding: (pad_h, pad_w)
            
        Returns:
            Output tensor (N, C_out, H_out, W_out)
        """
        # 1. parse the shape
        N, C_in, H_in, W_in = a.shape
        C_out, C_in2, KH, KW = b.shape
        assert C_in == C_in2, "Input channels must match weight channels"
        
        sh, sw = stride
        ph, pw = padding
        
        # 2. calculate the output shape
        H_out = (H_in + 2 * ph - KH) // sh + 1
        W_out = (W_in + 2 * pw - KW) // sw + 1
        
        out_shape = (N, C_out, H_out, W_out)
        out = a.zeros(out_shape)
        
        # a: input
        a_storage, a_shape_arr, a_strides = a._tensor._storage, a._tensor._shape, a._tensor._strides
        # b: weights
        b_storage, b_shape_arr, b_strides = b._tensor._storage, b._tensor._shape, b._tensor._strides
        # out: output
        out_storage, out_shape_arr, out_strides = out._tensor._storage, out._tensor._shape, out._tensor._strides
        
        out_index = np.zeros(4, dtype=np.int32)
        a_index = np.zeros(4, dtype=np.int32)
        b_index = np.zeros(4, dtype=np.int32)
        
        # 4. Naive Implementation
        # traverse the output position (n, c_out, h_out, w_out)
        for n in range(N):
            for c_out in range(C_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        
                        # calculate the output position
                        out_index[0], out_index[1], out_index[2], out_index[3] = n, c_out, h_out, w_out
                        out_pos = index_to_position(out_index, out_strides)
                        
                        acc = 0.0
                        
                        # sliding window: traverse the input channels and kernel size
                        for c_in in range(C_in):
                            for kh in range(KH):
                                for kw in range(KW):
                                    # calculate the corresponding coordinates on the input image (considering stride and padding)
                                    h_in = h_out * sh + kh - ph
                                    w_in = w_out * sw + kw - pw
                                    
                                    # boundary check (if padding causes out of range,视为 0)
                                    if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                        # construct the input index
                                        a_index[0], a_index[1], a_index[2], a_index[3] = n, c_in, h_in, w_in
                                        a_pos = index_to_position(a_index, a_strides)
                                        a_val = float(a_storage[a_pos])
                                        
                                        # construct the weight index (c_out, c_in, kh, kw)
                                        b_index[0], b_index[1], b_index[2], b_index[3] = c_out, c_in, kh, kw
                                        b_pos = index_to_position(b_index, b_strides)
                                        b_val = float(b_storage[b_pos])
                                        
                                        acc += a_val * b_val
                        
                        out_storage[out_pos] = acc

        return out

    @staticmethod
    def conv2d_backward(
        a: "Tensor",
        b: "Tensor",
        grad_out: "Tensor",
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> Tuple["Tensor", "Tensor"]:
 
        N, C_in, H_in, W_in = a.shape
        C_out, C_in2, KH, KW = b.shape
        assert C_in == C_in2, "Input channels must match weight channels"

        sh, sw = stride
        ph, pw = padding

        _, _, H_out, W_out = grad_out.shape

        d_a = a.zeros(a.shape)
        d_b = b.zeros(b.shape)

        a_storage, a_strides = a._tensor._storage, a._tensor._strides
        b_storage, b_strides = b._tensor._storage, b._tensor._strides
        go_storage, go_strides = grad_out._tensor._storage, grad_out._tensor._strides

        da_storage, da_strides = d_a._tensor._storage, d_a._tensor._strides
        db_storage, db_strides = d_b._tensor._storage, d_b._tensor._strides

        go_index = np.zeros(4, dtype=np.int32)
        a_index = np.zeros(4, dtype=np.int32)
        b_index = np.zeros(4, dtype=np.int32)
        # Begin Task 2.2
        # d_b[c_out, c_in, kh, kw] += sum_{n,h_out,w_out} a[n,c_in,h_in,w_in] * grad_out[n,c_out,h_out,w_out]
        # d_a[n, c_in, h_in, w_in] += sum_{c_out,kh,kw} b[c_out,c_in,kh,kw] * grad_out[n,c_out,h_out,w_out]
        for n in range(N):
            for c_out in range(C_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        # get grad_out value
                        go_idx = (n * go_strides[0] + 
                        c_out * go_strides[1] + 
                        h_out * go_strides[2] + 
                        w_out * go_strides[3])
                        go_val = go_storage[go_idx]
                        for c_in in range(C_in):
                            for kh in range(KH):
                                # compute h_in
                                h_in = h_out * sh + kh - ph                                
                                for kw in range(KW):
                                    # compute w_in
                                    w_in = w_out * sw + kw - pw
                                    if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                        # get a index / value and b index / value
                                        a_idx = (n * a_strides[0] + 
                                                c_in * a_strides[1] + 
                                                h_in * a_strides[2] + 
                                                w_in * a_strides[3])
                                        a_val = a_storage[a_idx]
                                        
                                        b_idx = (c_out * b_strides[0] + 
                                                c_in * b_strides[1] + 
                                                kh * b_strides[2] + 
                                                kw * b_strides[3])
                                        b_val = b_storage[b_idx]
                                        # accumulate d_a and d_b                                        
                                        db_idx = (c_out * db_strides[0] + 
                                                c_in * db_strides[1] + 
                                                kh * db_strides[2] + 
                                                kw * db_strides[3])
                                        db_storage[db_idx] += a_val * go_val
                                        
                                        da_idx = (n * da_strides[0] + 
                                                c_in * da_strides[1] + 
                                                h_in * da_strides[2] + 
                                                w_in * da_strides[3])
                                        da_storage[da_idx] += b_val * go_val

        return d_a, d_b
        # End Task 2.2


# Implementations.


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[TensorData, TensorData], None]:
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Fill in the `out` array by applying `fn` to each
    value of `in_storage` assuming `out_shape` and `in_shape`
    broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply

    Returns:
        Tensor map function.
    """

    def _map(out_data: TensorData, in_data: TensorData) -> None:
        """
        Apply `fn` elementwise, supporting broadcasting from `in_data` to `out_data`.
        """
        # Begin Task 1.3
        out_storage, out_shape, out_strides = out_data._storage, out_data._shape, out_data._strides
        in_storage, in_shape, in_strides = in_data._storage, in_data._shape, in_data._strides

        # Temporary index buffers
        out_index: Index = np.zeros(len(out_shape), dtype=np.int32)
        in_index: Index = np.zeros(len(in_shape), dtype=np.int32)
        # TODO:
        # 1. Traverse the output position.
        for i in range(out_data.size):
        # 2. Convert linear index to multi-d index for output.
            to_index(i,out_shape,out_index)# 线性索引 to 多维索引
        # 3. Broadcast output index to input index.
            broadcast_index(out_index,out_shape,in_shape,in_index)
        # 4. Convert multi-d index to position in storage.
            in_pos=index_to_position(in_index,in_strides) # to——index反过来
            out_pos=index_to_position(out_index,out_strides)
        # 5. Compute the output value with the input value using fn.
            val=in_storage[in_pos]
            result=fn(val)
            out_storage[out_pos]=result
        # For example, (1, 2) using neg: (-1, -2)

        #aise NotImplementedError("tensor_map not implemented")
        # End Task 1.3

    return _map


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [TensorData, TensorData, TensorData], None
]:
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Fill in the `out` array by applying `fn` to each
    value of `a_storage` and `b_storage` assuming `a_shape`
    and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply

    Returns:
        Tensor zip function.
    """

    def _zip(out_data: TensorData, a_data: TensorData, b_data: TensorData) -> None:
        """
        Apply `fn` elementwise to `a_data` and `b_data`, writing into `out_data`,
        with full broadcasting support.
        """
        # Begin Task 1.3
        out_storage, out_shape, out_strides = out_data._storage, out_data._shape, out_data._strides
        a_storage, a_shape, a_strides = a_data._storage, a_data._shape, a_data._strides
        b_storage, b_shape, b_strides = b_data._storage, b_data._shape, b_data._strides

        out_index: Index = np.zeros(len(out_shape), dtype=np.int32)
        a_index: Index = np.zeros(len(a_shape), dtype=np.int32)
        b_index: Index = np.zeros(len(b_shape), dtype=np.int32)

        # TODO:
        # 1. Traverse the output position
        # 2. Convert linear index to multi-d index for output.
        # 3. Broadcast output index to each input.
        # 4. Convert multi-d index to position in storage.
        # 5. Compute the output value with the input values using fn.
        # For example, (1, 2) and (3, 4) using add: (1+3, 2+4)

        for i in range(out_data.size):
        # 2. Convert linear index to multi-d index for output.
            to_index(i,out_shape,out_index)# 线性索引 to 多维索引
        # 3. Broadcast output index to input index.
            broadcast_index(out_index, out_shape, a_shape, a_index)  # 修正！
            broadcast_index(out_index, out_shape, b_shape, b_index)  # 修正！
        # 4. Convert multi-d index to position in storage.
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out_pos=index_to_position(out_index,out_strides)
        # 5. Compute the output value with the input value using fn.
            a_val=a_storage[a_pos]
            b_val=b_storage[b_pos]
            result=fn(a_val,b_val)
            out_storage[out_pos]=result
        # For example, (1, 2) using neg: (-1, -2)
            #print(f"out_shape={out_shape}, type={type(out_shape)}")
            #print(f"i={i}, type={type(i)}")
        #raise NotImplementedError("tensor_zip not implemented")
        # End Task 1.3
    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[TensorData, TensorData, int], None]:
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float

    Returns:
        Tensor reduce function.
    """

    def _reduce(out_data: TensorData, a_data: TensorData, reduce_dim: int,) -> None:
        """
        Reduce `a_data` along `reduce_dim` into `out_data` using binary op `fn`.
        `out_data` has the same number of dimensions as `a_data` but size 1 in
        the reduced dimension. Its storage should already be initialized with
        the identity element (e.g., 0 for sum, 1 for product).
        """
        # Begin Task 1.3
        out_storage, out_shape, out_strides = out_data._storage, out_data._shape, out_data._strides
        a_storage, a_shape, a_strides = a_data._storage, a_data._shape, a_data._strides

        out_index: Index = np.zeros(len(out_shape), dtype=np.int32)
        a_index: Index = np.zeros(len(a_shape), dtype=np.int32)

        # TODO:
        # 1. Traverse the output position.
        
        
        # For example, ((1, 2), (3, 4)) reducing on dim = 0 with add: ((1+3, 2+4))

        for i in range(out_data.size):
            to_index(i,out_shape,out_index)# 线性索引 to 多维索引,结果存在out_index中
        # 2. Initialize an accumulator from the CURRENT output value.
            out_pos=index_to_position(out_index,out_strides)
            accum=out_storage[out_pos]# out_storage已被初始化为幺元
        # 3. Convert linear index to multi-d index for output.
            for dim in range(len(a_shape)):
                if (dim==reduce_dim):
                    a_index[dim]=0
                else:
                    a_index[dim]=out_index[dim]
        # 4. Iterate over the reduction dimension with indices on other dimensions the same with output.
            reduce_size = a_shape[reduce_dim]
            for i in range(reduce_size):
                a_index[reduce_dim] = i
                a_pos = index_to_position(a_index, a_strides)
                a_val = a_storage[a_pos]
        # For each value, renew the accumulator using fn over the input value and the old value of accumulator.            
                accum = fn(accum, a_val) 
            out_storage[out_pos] = accum
        #raise NotImplementedError("tensor_reduce not implemented")
        # End Task 1.3
    return _reduce


SimpleBackend = TensorBackend(SimpleOps)