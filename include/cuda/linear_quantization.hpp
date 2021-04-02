/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/linear_quantization.h"

namespace mgard_cuda {

template <typename T>
void calc_quantizers(T *quantizers, int D, T norm, T tol, T s, int l_target,
                     bool reciprocal) {

  tol *= norm;
  tol *= 2;

  // original
  // tol /= l_target + 2;
  // for (int l = 0; l < l_target+1; l++) {
  //   quantizers[l] = tol;
  // }
  // printf("l_target %d\n", l_target);
  // levelwise
  // T C2 = 1 + 3*std::sqrt(3)/4;
  // T c = std::sqrt(std::pow(2, D));
  // T cc = (1 - c) / (1 - std::pow(c, l_target+1));
  // T level_eb = cc * tol / C2;

  // for (int l = 0; l < l_target+1; l++) {
  //   quantizers[l] = level_eb;
  //   level_eb *= c;
  // }

  // levelwise with s
  T C2 = 1 + 3 * std::sqrt(3) / 4;
  // T c = std::sqrt(std::pow(2, D));
  T c = std::sqrt(std::pow(2, D - 2 * s));
  T cc = (1 - c) / (1 - std::pow(c, l_target + 1));
  T level_eb = cc * tol / C2;

  for (int l = 0; l < l_target + 1; l++) {
    quantizers[l] = level_eb;
    // T c = std::sqrt(std::pow(2, 2*s*l + D * (l_target - l)));
    level_eb *= c;
    if (reciprocal)
      quantizers[l] = 1.0f / quantizers[l];
  }

  // print quantizers
  // printf("quantizers: ");
  // for (int l = 0; l < l_target+1; l++) {
  //   printf("%f ", quantizers[l]);
  // }
  // printf("\n");
}

template <typename T, int D, int R, int C, int F>
__global__ void
_levelwise_linear_quantize(int *shapes, int l_target, T *quantizers, T *dv,
                           int *ldvs, int *dwork, int *ldws, bool prep_huffmam,
                           int dict_size, int *shape, size_t *outlier_count,
                           unsigned int *outlier_idx, int *outliers) {

  size_t threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
                    (threadIdx.y * blockDim.x) + threadIdx.x;
  T *sm = SharedMemory<T>();
  T *quantizers_sm = sm;
  int *ldvs_sm = (int *)(quantizers_sm + l_target + 1);
  int *ldws_sm = ldvs_sm + D;
  int *shape_sm = ldws_sm + D;
  int *shapes_sm = shape_sm + D;

  if (threadId < l_target + 1) {
    quantizers_sm[threadId] = quantizers[threadId];
  }
  if (threadId < D) {
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
    shape_sm[threadId] = shape[threadId];
  }
  if (threadId < D * (l_target + 2)) {
    shapes_sm[threadId] = shapes[threadId];
    // printf ("D: %d l_target+2: %d load shapes[%llu]: %d\n", D, l_target+2,
    // threadId, shapes_sm[threadId]);
  }

  __syncthreads();

  int idx[D];
  int firstD = div_roundup(shapes_sm[l_target + 1], F);

  int bidx = blockIdx.x;
  idx[0] = (bidx % firstD) * F + threadIdx.x;

  // printf("shapes_sm[l_target+1]: %d firstD %d idx[0] %d\n",
  // shapes_sm[l_target+1], firstD, idx[0]);

  bidx /= firstD;
  if (D >= 2)
    idx[1] = blockIdx.y * blockDim.y + threadIdx.y;
  if (D >= 3)
    idx[2] = blockIdx.z * blockDim.z + threadIdx.z;

  for (int d = 3; d < D; d++) {
    idx[d] = bidx % shapes_sm[(l_target + 2) * d + l_target + 1];
    bidx /= shapes_sm[(l_target + 2) * d + l_target + 1];
  }

  int level = 0;
  for (int d = 0; d < D; d++) {
    long long unsigned int l_bit = 0l;
    for (int l = 0; l < l_target + 1; l++) {
      int bit = (idx[d] >= shapes_sm[(l_target + 2) * d + l]) &&
                (idx[d] < shapes_sm[(l_target + 2) * d + l + 1]);
      l_bit += bit << l;
      // printf("idx: %d %d d: %d l_bit: %llu\n", idx[1], idx[0], d, l_bit);
    }
    level = max(level, __ffsll(l_bit));
  }
  level = level - 1;

  bool in_range = true;
  for (int d = 0; d < D; d++) {
    if (idx[d] >= shapes_sm[(l_target + 2) * d + l_target + 1])
      in_range = false;
  }

  // printf("idx %llu, level: %d, in_range: %d idx[0]: shape_sm: %d\n",
  // get_idx<D>(shape_sm, idx), level, in_range, shapes_sm[(l_target+2) * 0 +
  // l_target+1]);

  if (level >= 0 && level <= l_target && in_range) {
    T t = dv[get_idx<D>(ldvs, idx)];
    int quantized_data = copysign(0.5 + fabs(t * quantizers_sm[level]), t);
    // printf("dv[%llu] %f quantizers[%d]%f -> dw[%llu]%d \n",
    //       get_idx<D>(ldvs, idx), t,
    //       level, quantizers_sm[level],
    //       get_idx<D>(ldws, idx), quantized_data+dict_size / 2);

    if (prep_huffmam) {
      quantized_data += dict_size / 2;
      if (quantized_data >= 0 && quantized_data < dict_size) {
        // do nothing
      } else {
        size_t i = atomicAdd((unsigned long long int *)outlier_count,
                             (unsigned long long int)1);
        outlier_idx[i] = (unsigned int)get_idx<D>(shape_sm, idx);
        outliers[i] = quantized_data;
        quantized_data = 0;
      }
      // if (get_idx<D>(shape_sm, idx) < quant_meta_size_ratio) {
      //   size_t i = atomicAdd((unsigned long long int*)outlier_count,
      //   (unsigned long long int)1); outlier_idx[i] = get_idx<D>(shape_sm,
      //   idx);
      // }
    }

    dwork[get_idx<D>(ldws, idx)] = quantized_data;
  }
}

template <typename T, int D, int R, int C, int F>
void levelwise_linear_quantize_adaptive_launcher(
    mgard_cuda_handle<T, D> &handle, int *shapes, int l_target, quant_meta<T> m,
    T *dv, int *ldvs, int *dwork, int *ldws, bool prep_huffmam, int *shape,
    size_t *outlier_count, unsigned int *outlier_idx, int *outliers,
    int queue_idx) {

  T *quantizers = new T[l_target + 1];
  calc_quantizers(quantizers, D, m.norm, m.tol, m.s, l_target, true);
  cudaMemcpyAsyncHelper(handle, handle.quantizers, quantizers,
                        sizeof(T) * (l_target + 1), H2D, queue_idx);
  // printf("norm: %f, tol: %f, s: %f, dict_size: %d\n", m.norm, m.tol, m.s,
  // m.dict_size);
  int total_thread_z = handle.dofs[2][0];
  int total_thread_y = handle.dofs[1][0];
  int total_thread_x = handle.dofs[0][0];
  // linearize other dimensions
  int tbz = R;
  int tby = C;
  int tbx = F;
  int gridz = ceil((float)total_thread_z / tbz);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  for (int d = 3; d < D; d++) {
    gridx *= handle.dofs[d][0];
  }

  // printf("exec: %d %d %d %d %d %d\n", tbx, tby, tbz, gridx, gridy, gridz);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  size_t sm_size = (D * 3) * sizeof(int);
  sm_size += (l_target + 1) * sizeof(T);
  sm_size += (l_target + 2) * D * sizeof(int);

  _levelwise_linear_quantize<T, D, R, C, F>
      <<<blockPerGrid, threadsPerBlock, sm_size,
         *(cudaStream_t *)handle.get(queue_idx)>>>(
          shapes, l_target, handle.quantizers, dv, ldvs, dwork, ldws,
          prep_huffmam, m.dict_size, shape, outlier_count, outlier_idx,
          outliers);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  // printf("DEBUG ON\n");
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template <typename T, int D>
void levelwise_linear_quantize(mgard_cuda_handle<T, D> &handle, int *shapes,
                               int l_target, quant_meta<T> m, T *dv, int *ldvs,
                               int *dwork, int *ldws, bool prep_huffmam,
                               int *shape, size_t *outlier_count,
                               unsigned int *outlier_idx, int *outliers,
                               int queue_idx) {
#define QUANTIZE(R, C, F)                                                      \
  {                                                                            \
    levelwise_linear_quantize_adaptive_launcher<T, D, R, C, F>(                \
        handle, shapes, l_target, m, dv, ldvs, dwork, ldws, prep_huffmam,      \
        shape, outlier_count, outlier_idx, outliers, queue_idx);               \
  }

  if (D >= 3) {
    QUANTIZE(4, 4, 16)
  }
  if (D == 2) {
    QUANTIZE(1, 4, 32)
  }
  if (D == 1) {
    QUANTIZE(1, 1, 64)
  }

#undef QUANTIZE
}

template <typename T, int D, int R, int C, int F>
__global__ void
_levelwise_linear_dequantize(int *shapes, int l_target, T *quantizers, int *dv,
                             int *ldvs, T *dwork, int *ldws, int dict_size,
                             size_t outlier_count, unsigned int *outlier_idx,
                             int *outliers) {

  size_t threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
                    (threadIdx.y * blockDim.x) + threadIdx.x;
  size_t blockId = (blockIdx.z * (gridDim.x * gridDim.y)) +
                   (blockIdx.y * gridDim.x) + blockIdx.x;
  size_t gloablId = blockId * blockDim.x * blockDim.y * blockDim.z + threadId;

  T *sm = SharedMemory<T>();
  T *quantizers_sm = sm;
  int *ldvs_sm = (int *)(quantizers_sm + l_target + 1);
  int *ldws_sm = ldvs_sm + D;
  int *shapes_sm = ldws_sm + D;

  if (threadId < l_target + 1) {
    quantizers_sm[threadId] = quantizers[threadId];
  }
  if (threadId < D) {
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
  }
  if (threadId < D * (l_target + 2)) {
    shapes_sm[threadId] = shapes[threadId];
  }

  __syncthreads();

  bool debug = false;
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    debug = true;
    // for (int d = 0; d < D; d ++) {
    //   printf("shapes_sm[%d]\n", d);
    //   for (int l = 0; l < l_target + 1; l++) {
    //     printf("%d ", shapes_sm[(l_target+1) * d + l]);
    //   }
    //   printf("\n");
    // }
  }

  __syncthreads();

  int idx[D];
  int firstD = div_roundup(shapes_sm[l_target + 1], F);

  // if (debug) printf("first n: %d\n", shapes_sm[l_target]);

  int bidx = blockIdx.x;
  idx[0] = (bidx % firstD) * F + threadIdx.x;

  // printf("firstD %d idx[0] %d\n", firstD, idx[0]);

  bidx /= firstD;
  if (D >= 2)
    idx[1] = blockIdx.y * blockDim.y + threadIdx.y;
  if (D >= 3)
    idx[2] = blockIdx.z * blockDim.z + threadIdx.z;

  for (int d = 3; d < D; d++) {
    idx[d] = bidx % shapes_sm[(l_target + 2) * d + l_target + 1];
    bidx /= shapes_sm[(l_target + 2) * d + l_target + 1];
  }

  int level = 0;
  for (int d = 0; d < D; d++) {
    long long unsigned int l_bit = 0l;
    for (int l = 0; l < l_target + 1; l++) {
      int bit = (idx[d] >= shapes_sm[(l_target + 2) * d + l]) &&
                (idx[d] < shapes_sm[(l_target + 2) * d + l + 1]);
      l_bit += bit << l;
      // printf("idx: %d %d d: %d l_bit: %llu\n", idx[1], idx[0], d, l_bit);
    }
    level = max(level, __ffsll(l_bit));
  }

  bool in_range = true;
  for (int d = 0; d < D; d++) {
    if (idx[d] >= shapes_sm[(l_target + 2) * d + l_target + 1])
      in_range = false;
  }

  level = level - 1;
  if (level >= 0 && level <= l_target && in_range) {
    // printf("%d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
    // printf("idx: %d %d l: %d\n", idx[1], idx[0], level);
    int quantized_data = dv[get_idx<D>(ldvs, idx)];
    quantized_data -= dict_size / 2;
    dwork[get_idx<D>(ldws, idx)] = quantizers_sm[level] * (T)quantized_data;
    // dwork[get_idx<D>(ldws, idx)] = (T)dv[get_idx<D>(ldvs, idx)];

    // printf("dw[%llu] %d dequantizers[%d]%f -> dw[%llu]%f \n",
    // get_idx<D>(ldvs, idx),
    //       quantized_data, level, quantizers_sm[level], get_idx<D>(ldws, idx),
    //       quantizers_sm[level] * (T)quantized_data);
  }

  // //outliers
  // if (gloablId < outlier_count) {
  //   size_t linerized_idx = outlier_idx[gloablId];
  //   for (int d = 0; d < D; d++) {
  //     idx[d] = linerized_idx % shapes_sm[(l_target+2) * d+l_target+1];
  //     linerized_idx /= shapes_sm[(l_target+2) * d+l_target+1];
  //   }
  //   int outliter = outliers[gloablId];
  //   outliter -= dict_size / 2;

  //   level = 0;
  //   for (int d = 0; d < D; d++) {
  //     long long unsigned int l_bit = 0l;
  //     for (int l = 0; l < l_target+1; l++) {
  //       int bit = (idx[d] >= shapes_sm[(l_target+2) * d + l]) && (idx[d] <
  //       shapes_sm[(l_target+2) * d + l+1]); l_bit += bit << l;
  //       // printf("idx: %d %d d: %d l_bit: %llu\n", idx[1], idx[0], d,
  //       l_bit);
  //     }
  //     level = max(level, __ffsll(l_bit));
  //   }
  //   level = level - 1;

  //   dwork[get_idx<D>(ldws, idx)] = quantizers_sm[level] * (T)outliter;

  //   // printf("outliter: dw[%llu] %d dequantizers[%d]%f -> dw[%llu]%f \n",
  //   get_idx<D>(ldvs, idx),
  //   //       outliter, level, quantizers_sm[level], get_idx<D>(ldws, idx),
  //   quantizers_sm[level] * (T)outliter);

  // }
}

template <typename T, int D, int R, int C, int F>
__global__ void _levelwise_linear_dequantize_outliers(
    int *shapes, int l_target, T *quantizers, int *dv, int *ldvs, T *dwork,
    int *ldws, int dict_size, size_t outlier_count, unsigned int *outlier_idx,
    int *outliers) {

  size_t threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
                    (threadIdx.y * blockDim.x) + threadIdx.x;
  size_t blockId = (blockIdx.z * (gridDim.x * gridDim.y)) +
                   (blockIdx.y * gridDim.x) + blockIdx.x;
  size_t gloablId = blockId * blockDim.x * blockDim.y * blockDim.z + threadId;

  T *sm = SharedMemory<T>();
  T *quantizers_sm = sm;
  int *ldvs_sm = (int *)(quantizers_sm + l_target + 1);
  int *ldws_sm = ldvs_sm + D;
  int *shapes_sm = ldws_sm + D;

  if (threadId < l_target + 1) {
    quantizers_sm[threadId] = quantizers[threadId];
  }
  if (threadId < D) {
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
  }
  if (threadId < D * (l_target + 2)) {
    shapes_sm[threadId] = shapes[threadId];
  }

  __syncthreads();
  int idx[D];

  // outliers
  if (gloablId < outlier_count) {
    size_t linerized_idx = outlier_idx[gloablId];
    for (int d = 0; d < D; d++) {
      idx[d] = linerized_idx % shapes_sm[(l_target + 2) * d + l_target + 1];
      linerized_idx /= shapes_sm[(l_target + 2) * d + l_target + 1];
    }
    int outliter = outliers[gloablId];
    outliter -= dict_size / 2;

    int level = 0;
    for (int d = 0; d < D; d++) {
      long long unsigned int l_bit = 0l;
      for (int l = 0; l < l_target + 1; l++) {
        int bit = (idx[d] >= shapes_sm[(l_target + 2) * d + l]) &&
                  (idx[d] < shapes_sm[(l_target + 2) * d + l + 1]);
        l_bit += bit << l;
        // printf("idx: %d %d d: %d l_bit: %llu\n", idx[1], idx[0], d, l_bit);
      }
      level = max(level, __ffsll(l_bit));
    }
    level = level - 1;

    dwork[get_idx<D>(ldws, idx)] = quantizers_sm[level] * (T)outliter;

    // printf("outliter: dw[%llu] %d dequantizers[%d]%f -> dw[%llu]%f \n",
    // get_idx<D>(ldvs, idx),
    //       outliter, level, quantizers_sm[level], get_idx<D>(ldws, idx),
    //       quantizers_sm[level] * (T)outliter);
  }
}

template <typename T, int D, int R, int C, int F>
void levelwise_linear_dequantize_adaptive_launcher(
    mgard_cuda_handle<T, D> &handle, int *shapes, int l_target, quant_meta<T> m,
    int *dv, int *ldvs, T *dwork, int *ldws, size_t outlier_count,
    unsigned int *outlier_idx, int *outliers, int queue_idx) {

  // printf("norm: %f, tol: %f, s: %f, dict_size: %d\n", m.norm, m.tol, m.s,
  // m.dict_size);

  T *quantizers = new T[l_target + 1];
  calc_quantizers(quantizers, D, m.norm, m.tol, m.s, l_target, false);
  cudaMemcpyAsyncHelper(handle, handle.quantizers, quantizers,
                        sizeof(T) * (l_target + 1), H2D, queue_idx);

  int total_thread_z = handle.dofs[2][0];
  int total_thread_y = handle.dofs[1][0];
  int total_thread_x = handle.dofs[0][0];
  // linearize other dimensions
  int tbz = R;
  int tby = C;
  int tbx = F;
  int gridz = ceil((float)total_thread_z / tbz);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  for (int d = 3; d < D; d++) {
    gridx *= handle.dofs[d][0];
  }

  // printf("exec: %d %d %d %d %d %d\n", tbx, tby, tbz, gridx, gridy, gridz);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  size_t sm_size = (D * 2) * sizeof(int);
  sm_size += (l_target + 1) * sizeof(T);
  sm_size += (l_target + 2) * D * sizeof(int);

  _levelwise_linear_dequantize<T, D, R, C, F>
      <<<blockPerGrid, threadsPerBlock, sm_size,
         *(cudaStream_t *)handle.get(queue_idx)>>>(
          shapes, l_target, handle.quantizers, dv, ldvs, dwork, ldws,
          m.dict_size, outlier_count, outlier_idx, outliers);
  _levelwise_linear_dequantize_outliers<T, D, R, C, F>
      <<<blockPerGrid, threadsPerBlock, sm_size,
         *(cudaStream_t *)handle.get(queue_idx)>>>(
          shapes, l_target, handle.quantizers, dv, ldvs, dwork, ldws,
          m.dict_size, outlier_count, outlier_idx, outliers);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template <typename T, int D>
void levelwise_linear_dequantize(mgard_cuda_handle<T, D> &handle, int *shapes,
                                 int l_target, quant_meta<T> m, int *dv,
                                 int *ldvs, T *dwork, int *ldws,
                                 size_t outlier_count,
                                 unsigned int *outlier_idx, int *outliers,
                                 int queue_idx) {
#define DEQUANTIZE(R, C, F)                                                    \
  {                                                                            \
    levelwise_linear_dequantize_adaptive_launcher<T, D, R, C, F>(              \
        handle, shapes, l_target, m, dv, ldvs, dwork, ldws, outlier_count,     \
        outlier_idx, outliers, queue_idx);                                     \
  }

  if (D >= 3) {
    DEQUANTIZE(4, 4, 16)
  }
  if (D == 2) {
    DEQUANTIZE(1, 4, 32)
  }
  if (D == 1) {
    DEQUANTIZE(1, 1, 64)
  }

#undef DEQUANTIZE
}

} // namespace mgard_cuda