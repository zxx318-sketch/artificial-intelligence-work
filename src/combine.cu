#include <cuda_runtime.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>

#define MAX_DIMS 10
#define TILE 32
#define BASE_THREAD_NUM 32

#define ADD_FUNC 1
#define MUL_FUNC 2
#define ID_FUNC 3
#define NEG_FUNC 4
#define LT_FUNC 5
#define EQ_FUNC 6
#define SIGMOID_FUNC 7
#define RELU_FUNC 8
#define RELU_BACK_FUNC 9
#define LOG_FUNC 10
#define LOG_BACK_FUNC 11
#define EXP_FUNC 12
#define INV_FUNC 13
#define INV_BACK_FUNC 14
#define IS_CLOSE_FUNC 15
#define MAX_FUNC 16
#define POW 17
#define TANH 18

__device__ float fn(int fn_id, float x, float y = 0)
{
  switch (fn_id)
  {
  case ADD_FUNC:
  {
    return x + y;
  }
  case MUL_FUNC:
  {
    return x * y;
  }
  case ID_FUNC:
  {
    return x;
  }
  case NEG_FUNC:
  {
    return -x;
  }
  case LT_FUNC:
  {
    if (x < y)
    {
      return 1.0;
    }
    else
    {
      return 0.0;
    }
  }
  case EQ_FUNC:
  {
    if (x == y)
    {
      return 1.0;
    }
    else
    {
      return 0.0;
    }
  }
  case SIGMOID_FUNC:
  {
    if (x >= 0)
    {
      return 1.0 / (1.0 + exp(-x));
    }
    else
    {
      return exp(x) / (1.0 + exp(x));
    }
  }
  case RELU_FUNC:
  {
    return max(x, 0.0);
  }
  case RELU_BACK_FUNC:
  {
    if (x > 0)
    {
      return y;
    }
    else
    {
      return 0.0;
    }
  }
  case LOG_FUNC:
  {
    return log(x + 1e-6);
  }
  case LOG_BACK_FUNC:
  {
    return y / (x + 1e-6);
  }
  case EXP_FUNC:
  {
    return exp(x);
  }
  case INV_FUNC:
  {
    return float(1.0 / x);
  }
  case INV_BACK_FUNC:
  {
    return -(1.0 / (x * x)) * y;
  }
  case IS_CLOSE_FUNC:
  {
    return (x - y < 1e-2) && (y - x < 1e-2);
  }
  case MAX_FUNC:
  {
    if (x > y)
    {
      return x;
    }
    else
    {
      return y;
    }
  }
  case POW:
  {
    return pow(x, y);
  }
  case TANH:
  {
    return tanh(x);
  }
  default:
  {
    return x + y;
  }
  }
}

__device__ int index_to_position(const int *index, const int *strides, int num_dims)
{
  int position = 0;
  for (int i = 0; i < num_dims; ++i)
  {
    position += index[i] * strides[i];
  }
  return position;
}

__device__ void to_index(int ordinal, const int *shape, int *out_index, int num_dims)
{
  int cur_ord = ordinal;
  for (int i = num_dims - 1; i >= 0; --i)
  {
    int sh = shape[i];
    out_index[i] = cur_ord % sh;
    cur_ord /= sh;
  }
}

__device__ void broadcast_index(const int *big_index, const int *big_shape, const int *shape, int *out_index, int num_dims_big, int num_dims)
{
  for (int i = 0; i < num_dims; ++i)
  {
    if (shape[i] > 1)
    {
      out_index[i] = big_index[i + (num_dims_big - num_dims)];
    }
    else
    {
      out_index[i] = 0;
    }
  }
}

__global__ void mapKernel(
    float *out,
    int *out_shape,
    int *out_strides,
    int out_size,
    float *in_storage,
    int *in_shape,
    int *in_strides,
    int shape_size,
    int fn_id)
{
  int out_index[MAX_DIMS];
  int in_index[MAX_DIMS];

  // TODO: 
  // Obtain the current thread's 1D global ID, map it to the logical coordinates (output index)
  // broadcast input index if needed
  // use fn_id to process the input value, and write the result to the output.
  
}

__global__ void reduceKernel(
    float *out,
    int *out_shape,
    int *out_strides,
    int out_size,
    float *a_storage,
    int *a_shape,
    int *a_strides,
    int reduce_dim,
    float reduce_value,
    int shape_size,
    int fn_id)
{
  // TODO: 
  // 1. Obtain the current thread's global ID (gid) and check if it is within the range of output elements.
  // 2. Use gid to compute the output tensor's logical multi-dimensional index.
  // 3. Broadcast out_index into the source input tensor in_index if needed (handle broadcasting rules).
  // 4. Convert in_index to the storage position for input, and out_index to output.
  // 5. Apply the appropriate elementwise function (selected by fn_id) on the input value(s) and store the result in output.
  
}

__global__ void zipKernel(
    float *out,
    int *out_shape,
    int *out_strides,
    int out_size,
    int out_shape_size,
    float *a_storage,
    int *a_shape,
    int *a_strides,
    int a_shape_size,
    float *b_storage,
    int *b_shape,
    int *b_strides,
    int b_shape_size,
    int fn_id)
{
  // TODO:
  // 1. Get the current thread's global id (gid) and make sure it's within the output tensor out_size.
  // 2. Compute the multi-dimensional output index.
  // 3. Broadcast out_index to a_index and b_index according to broadcasting rules:
  //    - Align dimensions from the right; if a dimension is 1, always use index 0, otherwise use the value from out_index.
  // 4. Convert logical indices to memory positions.
  // 5. Read data from a_storage and b_storage, combine them using fn(fn_id, ...), and write the result back.

}

__global__ void MatrixMultiplyKernel(
    float *out,
    const int *out_shape,
    const int *out_strides,
    float *a_storage,
    const int *a_shape,
    const int *a_strides,
    float *b_storage,
    const int *b_shape,
    const int *b_strides)
{
  // TODO

  // 1. Determine thread's logical coordinates and matrix boundaries
  //    Use blockIdx and threadIdx to locate the thread's assignment

  // 2. Allocate shared memory and initialize accumulator
  //    Allocate TILE x TILE shared memory As/Bs for caching, and local register c_value for accumulation
  // (This is handled after this insertion point via __shared__ arrays and c_value initialization)

  // 3. Sliding window loop on the K dimension
  //    Iterate k_offset with step TILE, process multiple tiles (blocks) of matrix multiplication

  // 4. Cooperative loading of data blocks (Global Memory -> Shared Memory)
  //    Each thread loads *one* element for a and b block; pad with zeros if out of range

  // 5. First thread synchronization
  //    Use __syncthreads() to ensure all threads have loaded data into shared mem before compute

  // 6. Compute tile's local product (Compute)
  //    Each thread computes on the shared memory As, Bs, accumulates result in c_value

  // 7. Second thread synchronization
  //    Second __syncthreads() before proceeding to next tile of K to avoid overwriting shared mem

  // 8. Eliminate invalid threads and write results (Shared Mem -> Global Mem)
  //    After all tiles, only legal threads write their result back to out using out_strides for index
  
}

__global__ void im2colKernel(
    float *out, const int *out_strides,
    const float *in, const int *in_shape, const int *in_strides,
    int N, int C, int H, int W,
    int Kh, int Kw,
    int Sh, int Sw,
    int Oh, int Ow)
{
  /*
  Each thread computes one element in the output matrix out[row, col].
  */
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int rows = C * Kh * Kw;
  int cols = N * Oh * Ow;
  int total = rows * cols;
  if (gid >= total)
    return;

  int row = gid / cols;
  int col = gid % cols;

  int kw = row % Kw;
  int tmp = row / Kw;
  int kh = tmp % Kh;
  int c = tmp / Kh;

  int ow = col % Ow;
  tmp = col / Ow;
  int oh = tmp % Oh;
  int n = tmp / Oh;

  int h = oh * Sh + kh;
  int w = ow * Sw + kw;

  float v = 0.0f;
  if (h >= 0 && h < H && w >= 0 && w < W)
  {
    int in_pos = n * in_strides[0] + c * in_strides[1] + h * in_strides[2] + w * in_strides[3];
    v = in[in_pos];
  }

  int out_pos = row * out_strides[0] + col * out_strides[1];
  out[out_pos] = v;
}

__global__ void col2imKernel(
    float *out, const int *out_shape, const int *out_strides,
    const float *in,
    int N, int C, int H, int W,
    int Kh, int Kw,
    int Sh, int Sw,
    int Oh, int Ow)
{
  /*
  Each thread reads one element in col matrix in[row, col] and atomicAdd it to the corresponding position in the output.
  */
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int rows = C * Kh * Kw;
  int cols = N * Oh * Ow;
  int total = rows * cols;
  if (gid >= total)
    return;

  int row = gid / cols;
  int col = gid % cols;

  int kw = row % Kw;
  int tmp = row / Kw;
  int kh = tmp % Kh;
  int c = tmp / Kh;

  int ow = col % Ow;
  tmp = col / Ow;
  int oh = tmp % Oh;
  int n = tmp / Oh;

  int h = oh * Sh + kh;
  int w = ow * Sw + kw;
  if (h < 0 || h >= H || w < 0 || w >= W)
    return;

  float v = in[row * (cols) + col];
  int out_pos = n * out_strides[0] + c * out_strides[1] + h * out_strides[2] + w * out_strides[3];
  atomicAdd(out + out_pos, v);
}

extern "C"
{

  void MatrixMultiply(
      float *out,
      int *out_shape,
      int *out_strides,
      float *a_storage,
      int *a_shape,
      int *a_strides,
      float *b_storage,
      int *b_shape,
      int *b_strides,
      int batch, int m, int p)
  {
    int n = a_shape[2];

    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaMalloc(&d_a, batch * m * n * sizeof(float));
    cudaMalloc(&d_b, batch * n * p * sizeof(float));
    cudaMalloc(&d_out, batch * m * p * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    cudaMalloc(&d_out_shape, 3 * sizeof(int));
    cudaMalloc(&d_out_strides, 3 * sizeof(int));
    cudaMalloc(&d_a_shape, 3 * sizeof(int));
    cudaMalloc(&d_a_strides, 3 * sizeof(int));
    cudaMalloc(&d_b_shape, 3 * sizeof(int));
    cudaMalloc(&d_b_strides, 3 * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_a, a_storage, batch * m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, batch * n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = BASE_THREAD_NUM;
    dim3 blockDims(threadsPerBlock, threadsPerBlock, 1); // Adjust these values based on your specific requirements
    dim3 gridDims((m + threadsPerBlock - 1) / threadsPerBlock, (p + threadsPerBlock - 1) / threadsPerBlock, batch);
    MatrixMultiplyKernel<<<gridDims, blockDims>>>(
        d_out, d_out_shape, d_out_strides, d_a, d_a_shape, d_a_strides, d_b, d_b_shape, d_b_strides);

    // Copy back to the host
    cudaMemcpy(out, d_out, batch * m * p * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Matmul Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
  }

  void tensorMap(
      float *out,
      int *out_shape,
      int *out_strides,
      int out_size,
      float *in_storage,
      int *in_shape,
      int *in_strides,
      int in_size,
      int shape_size,
      int fn_id)
  {

    float *d_out, *d_in;
    cudaMalloc(&d_out, out_size * sizeof(float));
    cudaMalloc(&d_in, in_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_in_shape, *d_in_strides;
    cudaMalloc(&d_out_shape, shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, shape_size * sizeof(int));
    cudaMalloc(&d_in_shape, shape_size * sizeof(int));
    cudaMalloc(&d_in_strides, shape_size * sizeof(int));

    cudaMemcpy(d_in, in_storage, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_shape, in_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, in_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    mapKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out, d_out_shape, d_out_strides, out_size,
        d_in, d_in_shape, d_in_strides,
        shape_size, fn_id);

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Map Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_in_shape);
    cudaFree(d_in_strides);
  }

  void tensorZip(
      float *out,
      int *out_shape,
      int *out_strides,
      int out_size,
      int out_shape_size,
      float *a_storage,
      int *a_shape,
      int *a_strides,
      int a_size,
      int a_shape_size,
      float *b_storage,
      int *b_shape,
      int *b_strides,
      int b_size,
      int b_shape_size,
      int fn_id)
  {

    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaMalloc((void **)&d_a, a_size * sizeof(float));
    cudaMalloc(&d_b, b_size * sizeof(float));
    cudaMalloc(&d_out, out_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    cudaMalloc(&d_out_shape, out_shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, out_shape_size * sizeof(int));
    cudaMalloc(&d_a_shape, a_shape_size * sizeof(int));
    cudaMalloc(&d_a_strides, a_shape_size * sizeof(int));
    cudaMalloc(&d_b_shape, b_shape_size * sizeof(int));
    cudaMalloc(&d_b_strides, b_shape_size * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_a, a_storage, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, b_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, out_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, out_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, a_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, a_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, b_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, b_shape_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    zipKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out, d_out_shape, d_out_strides, out_size, out_shape_size,
        d_a, d_a_shape, d_a_strides, a_shape_size,
        d_b, d_b_shape, d_b_strides, b_shape_size,
        fn_id);

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Zip Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
  }

  void tensorReduce(
      float *out,
      int *out_shape,
      int *out_strides,
      int out_size,
      float *a_storage,
      int *a_shape,
      int *a_strides,
      int reduce_dim,
      float reduce_value,
      int shape_size,
      int fn_id)
  {
    int a_size = out_size * a_shape[reduce_dim];
    float *d_out, *d_a;
    cudaMalloc(&d_out, out_size * sizeof(float));
    cudaMalloc(&d_a, a_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides;
    cudaMalloc(&d_out_shape, shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, shape_size * sizeof(int));
    cudaMalloc(&d_a_shape, shape_size * sizeof(int));
    cudaMalloc(&d_a_strides, shape_size * sizeof(int));

    cudaMemcpy(d_a, a_storage, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out, d_out_shape, d_out_strides, out_size,
        d_a, d_a_shape, d_a_strides,
        reduce_dim, reduce_value, shape_size, fn_id);

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Reduce Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
  }

  extern "C" void im2colLauncher(
      float *out,
      int *out_shape,
      int *out_strides,
      float *in,
      int *in_shape,
      int *in_strides,
      int N, int C, int H, int W,
      int Kh, int Kw, int Sh, int Sw,
      int Oh, int Ow)
  {
    int rows = C * Kh * Kw;
    int cols = N * Oh * Ow;
    int total_elems = rows * cols;

    float *d_out, *d_in;
    cudaMalloc(&d_out, total_elems * sizeof(float));
    cudaMalloc(&d_in, N * C * H * W * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_in_shape, *d_in_strides;
    cudaMalloc(&d_out_shape, 2 * sizeof(int));
    cudaMalloc(&d_out_strides, 2 * sizeof(int));
    cudaMalloc(&d_in_shape, 4 * sizeof(int));
    cudaMalloc(&d_in_strides, 4 * sizeof(int));

    cudaMemcpy(d_in, in, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_shape, in_shape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, in_strides, 4 * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (total_elems + threadsPerBlock - 1) / threadsPerBlock;
    im2colKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out, d_out_strides, d_in, d_in_shape, d_in_strides,
        N, C, H, W, Kh, Kw, Sh, Sw, Oh, Ow);

    // Copy back to the host
    cudaMemcpy(out, d_out, total_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      fprintf(stderr, "im2col Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_in_shape);
    cudaFree(d_in_strides);
  }

  extern "C" void col2imLauncher(
      float *out,
      int *out_shape,
      int *out_strides,
      float *in,
      int *in_shape,
      int *in_strides,
      int N, int C, int H, int W,
      int Kh, int Kw, int Sh, int Sw,
      int Oh, int Ow)
  {
    int rows = C * Kh * Kw;
    int cols = N * Oh * Ow;
    int total_elems = rows * cols;
    int out_size = N * C * H * W * sizeof(float);
    int in_size = total_elems * sizeof(float);

    float *d_out, *d_in;
    cudaMalloc(&d_out, out_size);
    cudaMalloc(&d_in, in_size);

    int *d_out_shape, *d_out_strides, *d_in_shape, *d_in_strides;
    cudaMalloc(&d_out_shape, 4 * sizeof(int));
    cudaMalloc(&d_out_strides, 4 * sizeof(int));
    cudaMalloc(&d_in_shape, 2 * sizeof(int));
    cudaMalloc(&d_in_strides, 2 * sizeof(int));

    // Initialize output to zero on device before atomicAdd!
    cudaMemset(d_out, 0, out_size);

    cudaMemcpy(d_in, in, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_shape, in_shape, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, in_strides, 2 * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = BASE_THREAD_NUM;
    int blocksPerGrid = (total_elems + threadsPerBlock - 1) / threadsPerBlock;

    col2imKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out, d_out_shape, d_out_strides, d_in,
        N, C, H, W, Kh, Kw, Sh, Sw, Oh, Ow);

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      fprintf(stderr, "col2im Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_in_shape);
    cudaFree(d_in_strides);
  }
}