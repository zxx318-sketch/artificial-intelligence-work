from typing import Callable, Optional, Tuple

from . import operators
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps
from .tensor_functions import tensor_from_numpy

import ctypes
import numpy as np
import os
from .tensor_ops import TensorBackend

# Load the shared library
lib = None
try:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == "win32":
        lib_name = "combine.dll"
    else:
        lib_name = "combine.so"
    lib_path = os.path.join(current_dir, "cuda_kernels", lib_name)
    lib = ctypes.CDLL(lib_path)
except Exception as e:
    print(f"cuda kernels not implemented: {lib_name} not found ({e})")
    if sys.platform == "win32":
        print("  [HINT] If you have an NVIDIA GPU, run 'compile_cuda.bat' to build the CUDA kernels.")
        print("  [HINT] Make sure CUDA Toolkit 'bin' folder is in your PATH.")

datatype = np.float32

# function map
fn_map = {
  operators.add: 1,
  operators.mul: 2,
  operators.id: 3,
  operators.neg: 4,
  operators.lt: 5,
  operators.eq: 6,
  operators.sigmoid: 7,
  operators.relu: 8,
  operators.relu_back: 9,
  operators.log: 10,
  operators.log_back: 11,
  operators.exp: 12,
  operators.inv: 13,
  operators.inv_back: 14,
  operators.is_close: 15,
  operators.max: 16,
  operators.pow: 17,
  operators.tanh: 18,
  operators.leaky_relu: 19,
  operators.leaky_relu_back: 20,
}

THREADS_PER_BLOCK = 32

class CudaKernelOps(TensorOps):
    # This backend uses ctypes + CUDA kernels with explicit host/device copies.
    # Keep Tensor storage as NumPy arrays (do not auto-convert to numba device arrays).
    cuda = False
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        fn_id = fn_map[fn]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            lib.tensorMap.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # in_size
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorMap.restype = None
            
            # assert out.size == a.size, f"zip {out.size}, {a.size}"

            lib.tensorMap(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            lib.tensorZip.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                ctypes.c_int,                                                            # out_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_strides
                ctypes.c_int,                                                            # a_size
                ctypes.c_int,                                                            # a_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # b_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_strides
                ctypes.c_int,                                                            # b_size
                ctypes.c_int,                                                            # b_shape_size
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorZip.restype = None

            # assert out.size == a.size, f"zip {out.size}, {a.size}"
            # assert out.size == b.size, f"zip {out.size}, {b.size}"

            lib.tensorZip(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                len(out.shape),
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                b._tensor._storage,
                b._tensor._shape.astype(np.int32),
                b._tensor._strides.astype(np.int32),
                b.size,
                len(b.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0) -> Callable[[Tensor, int], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))

            lib.tensorReduce.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # reduce_dim
                ctypes.c_float,                                                         # reduce_value
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorReduce.restype = None

            lib.tensorReduce(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                dim,
                start,
                len(a.shape),
                fn_id
            )

            return out

        return ret


    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # handle cases with more dimensions [64, 4, 32, 128] x [64, 4, 128, 32]
        more_3d = False
        if len(out.shape) > 3:
            # print(f"Debug in matmul: output shape {ls}")
            more_3d = True
            out = out.view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])
            nshape = out._tensor._shape
            nstrides = out._tensor._strides
            # print(f"Debug in matmul: batched dim [:-2] and get the strides {nshape, nstrides}")
        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2], a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2], b.shape[-1])
        
        assert a.shape[0] == b.shape[0]
        assert a.shape[0] == out.shape[0]

        lib.MatrixMultiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # b_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_strides
            ctypes.c_int,                                                             # batch_size
            ctypes.c_int,                                                             # out_shape[1], m
            ctypes.c_int                                                              # out_shape[2], p
        ]

        lib.MatrixMultiply.restype = None

        assert len(out._tensor._shape) == 3, f"{len(out._tensor._shape)}"
        assert len(out._tensor._strides) == 3, f"{len(out._tensor._strides)}"
        assert len(a._tensor._shape) == 3
        assert len(a._tensor._strides) == 3
        assert len(b._tensor._shape) == 3
        assert len(b._tensor._strides) == 3

        lib.MatrixMultiply(
            out._tensor._storage,
            out._tensor._shape.astype(np.int32),
            out._tensor._strides.astype(np.int32),
            a._tensor._storage,
            a._tensor._shape.astype(np.int32),
            a._tensor._strides.astype(np.int32),
            b._tensor._storage,
            b._tensor._shape.astype(np.int32),
            b._tensor._strides.astype(np.int32),
            a.shape[0],
            a.shape[1],
            b.shape[2]
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        if more_3d:
            out = out.view(*ls)
            # print(f"Debug in matmul: output shape {out.shape}")
        return out

    @staticmethod
    def im2col(a: Tensor, kernel_shape: Tuple[int, int], stride: Tuple[int, int]) -> Tensor:
        N, C, H, W = a.shape
        Kh, Kw = kernel_shape
        Sh, Sw = stride
        
        Oh = (H - Kh) // Sh + 1
        Ow = (W - Kw) // Sw + 1
        
        out_shape = (C * Kh * Kw, N * Oh * Ow)
        out = a.zeros(out_shape)
        
        lib.im2colLauncher.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # out_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'), # out_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'), # out_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # in_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'), # in_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'), # in_strides
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, # N, C, H, W
            ctypes.c_int, ctypes.c_int, # Kh, Kw
            ctypes.c_int, ctypes.c_int, # Sh, Sw
            ctypes.c_int, ctypes.c_int  # Oh, Ow
        ]
        lib.im2colLauncher.restype = None

        lib.im2colLauncher(
            out._tensor._storage,
            out._tensor._shape.astype(np.int32),
            out._tensor._strides.astype(np.int32),
            a._tensor._storage,
            a._tensor._shape.astype(np.int32),
            a._tensor._strides.astype(np.int32),
            N, C, H, W,
            Kh, Kw,
            Sh, Sw,
            Oh, Ow
        )
        
        return out

    @staticmethod
    def col2im(a: Tensor, output_shape: Tuple[int, ...], kernel_shape: Tuple[int, int], stride: Tuple[int, int]) -> Tensor:
        N, C, H, W = output_shape
        Kh, Kw = kernel_shape
        Sh, Sw = stride
        
        Oh = (H - Kh) // Sh + 1
        Ow = (W - Kw) // Sw + 1
        
        out = a.zeros(output_shape)
        
        lib.col2imLauncher.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # out_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'), # out_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'), # out_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # in_storage (grad_col)
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'), # in_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'), # in_strides
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, # N, C, H, W
            ctypes.c_int, ctypes.c_int, # Kh, Kw
            ctypes.c_int, ctypes.c_int, # Sh, Sw
            ctypes.c_int, ctypes.c_int  # Oh, Ow
        ]
        lib.col2imLauncher.restype = None

        # 注意：col2im 内部必须使用 atomicAdd 因为多个输出位置可能映射到同一个输入像素
        lib.col2imLauncher(
            out._tensor._storage,
            out._tensor._shape.astype(np.int32),
            out._tensor._strides.astype(np.int32),
            a._tensor._storage,
            a._tensor._shape.astype(np.int32),
            a._tensor._strides.astype(np.int32),
            N, C, H, W,
            Kh, Kw,
            Sh, Sw,
            Oh, Ow
        )
        
        return out
    
    @staticmethod
    def conv2d(a: Tensor, b: Tensor, stride: Tuple[int, int], padding: Tuple[int, int]) -> Tensor:
        """
        Args:
            a: Input tensor (N, C_in, H_in, W_in)
            b: Weight tensor (C_out, C_in, KH, KW)
            stride: (stride_h, stride_w)
            padding: (pad_h, pad_w)
            
        Returns:
            Output tensor (N, C_out, H_out, W_out)
        """
        ph, pw = padding
        a_work = a
        if ph > 0 or pw > 0:
            a_np = a.to_numpy()
            a_pad = np.pad(a_np, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
            a_work = tensor_from_numpy(a_pad, backend=a.backend)

        # im2col: (C_in * KH * KW, N * H_out * W_out)
        col = a_work.f.im2col(a_work, b.shape[-2:], stride)

        # weight reshape: (C_out, C_in * KH * KW)
        w = b.contiguous().view(b.shape[0], b.shape[1] * b.shape[2] * b.shape[3])

        # batched matmul expects 3D tensors: [B, M, K] x [B, K, P] -> [B, M, P]
        w3 = w.contiguous().view(1, w.shape[0], w.shape[1])
        col3 = col.contiguous().view(1, col.shape[0], col.shape[1])
        out = a.f.matrix_multiply(w3, col3).contiguous().view(w.shape[0], col.shape[1])

        # reshape to NCHW
        N, _, H, W = a_work.shape
        KH, KW = b.shape[-2], b.shape[-1]
        SH, SW = stride
        OH = (H - KH) // SH + 1
        OW = (W - KW) // SW + 1

        out = out.contiguous().view(b.shape[0], N, OH, OW).permute(1, 0, 2, 3).contiguous()
        return out
    @staticmethod
    def conv2d_backward(
        a: "Tensor",
        b: "Tensor",
        grad_out: "Tensor",
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> Tuple["Tensor", "Tensor"]:
        """
        Efficient CUDA-style conv2d backward using im2col and col2im.
        Handles padding cropping via NumPy since Tensor slicing is not supported.
        """
        ph, pw = padding
        sh, sw = stride

        N, C_in, H_in, W_in = a.shape
        C_out, _, KH, KW = b.shape
        _, _, OH, OW = grad_out.shape

        # 1. 对输入 a 进行与正向传播完全一样的 Padding 操作
        a_work = a
        if ph > 0 or pw > 0:
            a_np = a.to_numpy()
            a_pad = np.pad(a_np, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
            a_work = tensor_from_numpy(a_pad, backend=a.backend)

        # im2col: (C_in * KH * KW, N * OH * OW)
        col = a_work.f.im2col(a_work, (KH, KW), stride)
        # 转置 X_col: (N * OH * OW, C_in * KH * KW)
        col_T = col.permute(1, 0).contiguous()
        
        # (C_out, N, OH, OW) -> (C_out, N * OH * OW)
        grad_out_flat = grad_out.permute(1, 0, 2, 3).contiguous().view(C_out, N * OH * OW)
        
        
        g3 = grad_out_flat.contiguous().view(1, C_out, N * OH * OW)
        col3_T = col_T.contiguous().view(1, N * OH * OW, C_in * KH * KW)
        
        # (1, C_out, N*OH*OW) @ (1, N*OH*OW, C_in*KH*KW) -> (1, C_out, C_in*KH*KW)
        dW_flat3 = grad_out_flat.f.matrix_multiply(g3, col3_T)
        
        d_weight = dW_flat3.contiguous().view(C_out, C_in, KH, KW) # (C_out, C_in, KH, KW)
        
        
        w_flat = b.contiguous().view(C_out, C_in * KH * KW) # flatten W: (C_out, C_in * KH * KW)
        w_flat_T = w_flat.permute(1, 0).contiguous() # (C_in * KH * KW, C_out)
        
        w3_T = w_flat_T.contiguous().view(1, C_in * KH * KW, C_out)
        
        # (1, C_in*KH*KW, C_out) @ (1, C_out, N*OH*OW) -> (1, C_in*KH*KW, N*OH*OW)
        dX_col3 = grad_out_flat.f.matrix_multiply(w3_T, g3)
        
        
        dX_col = dX_col3.contiguous().view(C_in * KH * KW, N * OH * OW) # (C_in * KH * KW, N * OH * OW)
        
        # col2im （含 Padding)
        d_input_padded = dX_col.f.col2im(dX_col, a_work.shape, (KH, KW), stride) # (N, C_in, H_in + 2*ph, W_in + 2*pw)

        # 去除 Padding
        # 将 Tensor 抽回 NumPy -> 在 NumPy 里切片 -> 再推回 Tensor
        # =====================================================================
        if ph > 0 or pw > 0:
            dip_np = d_input_padded.to_numpy()
            # 在 Numpy 中进行安全的高维切片
            dip_cropped = dip_np[:, :, ph:ph+H_in, pw:pw+W_in]
            d_input = tensor_from_numpy(dip_cropped, backend=a.backend)
        else:
            d_input = d_input_padded

        return d_input, d_weight
        

CUDABackend = TensorBackend(CudaKernelOps)