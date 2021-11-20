/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */
#include <stdint.h>

#include <algorithm>
#include <cstdio>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>



#ifndef MGRAD_CUDA_COMMON_INTERNAL
#define MGRAD_CUDA_COMMON_INTERNAL

#if defined(__CUDACC__) // NVCC
   #define MGARDm_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define MGARDm_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define MGARDm_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif


#define MGARDm_CONT __host__  __inline__
#define MGARDm_KERL __global__
#define MGARDm_EXEC __device__ __forceinline__
#define MGARDm_CONT_EXEC __host__ __device__ __forceinline__
#define MGARDm_COMPILE_EXEC __CUDACC__

#include "Common.h"

#define MAX_GRID_X 2147483647
#define MAX_GRID_Y 65536
#define MAX_GRID_Z 65536
#define MGARDm_WARP_SIZE 32
#define MGARDm_MAX_NUM_WARPS_PER_TB 32
#define MGARDm_NUM_SMs 68

#define MGARDm_NUM_QUEUES 16

#define SIZE_MAX_VALUE 4294967295

#define COPY 0
#define ADD 1
#define SUBTRACT 2

// reduction operations
#define SUM 0
#define MAX 1

// #define WARP_SIZE 32
// #define ROUND_UP_WARP(TID) ((TID) + WARP_SIZE - 1) / WARP_SIZE

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

namespace mgard_cuda {


template <class T> struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

template <DIM D> int check_shape(std::vector<SIZE> shape) {
  if (D != shape.size()) {
    return -1;
  }
  for (DIM i = 0; i < shape.size(); i++) {
    if (shape[i] < 3)
      return -2;
  }
  return 0;
}

bool is_2kplus1_cuda(double num);

// __device__ int get_idx(const int ld, const int i, const int j);

// __device__ int get_idx(const int ld1, const int ld2, const int i, const int
// j,
//                        const int k);

// __forceinline__ __device__ int get_idx(const int ld, const int i, const int j) {
//   return ld * i + j;
// }

// ld2 = nrow
// ld1 = pitch
// for 1-3D
__host__ __forceinline__ __device__ LENGTH
get_idx(const SIZE ld1, const SIZE ld2, const SIZE z, const SIZE y, const SIZE x) {
  return ld2 * ld1 * z + ld1 * y + x;
}

// for 3D+
__host__ __forceinline__ __device__ LENGTH
get_idx(const LENGTH ld1, const LENGTH ld2, const SIZE z, const SIZE y, const SIZE x) {
  return ld2 * ld1 * z + ld1 * y + x;
}

// leading dimension first
__host__ inline LENGTH get_idx(std::vector<SIZE> lds, std::vector<SIZE> idx) {
  LENGTH curr_stride = 1;
  LENGTH ret_idx = 0;
  for (DIM i = 0; i < idx.size(); i++) {
    ret_idx += idx[i] * curr_stride;
    curr_stride *= lds[i];
  }
  return ret_idx;
}

template <DIM D> __forceinline__ __device__ LENGTH get_idx(SIZE *lds, SIZE *idx) {
  LENGTH curr_stride = 1;
  LENGTH ret_idx = 0;
  for (DIM i = 0; i < D; i++) {
    ret_idx += idx[i] * curr_stride;
    curr_stride *= lds[i];
  }
  return ret_idx;
}

__host__ inline std::vector<SIZE> gen_idx(DIM D, DIM curr_dim_r, DIM curr_dim_c,
                                         DIM curr_dim_f, SIZE idx_r, SIZE idx_c,
                                         SIZE idx_f) {
  std::vector<SIZE> idx(D, 0);
  idx[curr_dim_r] = idx_r;
  idx[curr_dim_c] = idx_c;
  idx[curr_dim_f] = idx_f;
  return idx;
}

__host__ __forceinline__ __device__ int div_roundup(SIZE a, SIZE b) {
  return (a - 1) / b + 1;
}

template<typename T1, typename T2>
__host__ __forceinline__ __device__ SIZE roundup(T2 a) {
  return ((a - 1) / sizeof(T1) + 1) * sizeof(T1);
}

// template <int D, int R, int C, int F>
// __host__ inline void kernel_config(thrust::device_vector<int> &shape, int &tbx,
//                                    int &tby, int &tbz, int &gridx, int &gridy,
//                                    int &gridz,
//                                    thrust::device_vector<int> &assigned_dimx,
//                                    thrust::device_vector<int> &assigned_dimy,
//                                    thrust::device_vector<int> &assigned_dimz) {

//   tbx = F;
//   tby = C;
//   tbz = R;
//   gridx = ceil((double)shape[0] / F);
//   gridy = ceil((double)shape[1] / C);
//   gridz = ceil((double)shape[2] / R);
//   assigned_dimx.push_back(0);
//   assigned_dimy.push_back(1);
//   assigned_dimz.push_back(2);

//   int d = 3;
//   while (d < D) {
//     if (gridx * shape[d] < MAX_GRID_X) {
//       gridx *= shape[d];
//       assigned_dimx.push_back(d);
//       d++;
//     } else {
//       break;
//     }
//   }

//   while (d < D) {
//     if (gridy * shape[d] < MAX_GRID_Y) {
//       gridy *= shape[d];
//       assigned_dimy.push_back(d);
//       d++;
//     } else {
//       break;
//     }
//   }

//   while (d < D) {
//     if (gridz * shape[d] < MAX_GRID_Z) {
//       gridz *= shape[d];
//       assigned_dimz.push_back(d);
//       d++;
//     } else {
//       break;
//     }
//   }
// }

// template <int D, int R, int C, int F>
// __forceinline__ __device__ void
// get_idx(int *shape, int assigned_nx, int *assigned_dimx, int assigned_ny,
//         int *assigned_dimy, int assigned_nz, int *assigned_dimz, int *idx) {
//   int bidx = blockIdx.x;
//   int bidy = blockIdx.y;
//   int bidz = blockIdx.z;
//   idx[0] = (bidx % shape[0]) * F + threadIdx.x;
//   idx[1] = (bidy % shape[1]) * C + threadIdx.y;
//   idx[2] = (bidz % shape[2]) * R + threadIdx.z;
//   if (idx[0] < 0) {
//     printf("neg %d %d %d %d\n", bidx, shape[0], F, threadIdx.x);
//   }
//   if (idx[1] < 0) {
//     printf("neg %d %d %d %d\n", bidy, shape[1], C, threadIdx.y);
//   }
//   if (idx[2] < 0) {
//     printf("neg %d %d %d %d\n", bidz, shape[2], R, threadIdx.z);
//   }
//   // bidx /= shape[0];
//   // bidy /= shape[1];
//   // bidz /= shape[2];
//   // for (int i = 1; i < assigned_nx; i++) {
//   //   int d = assigned_dimx[i];
//   //   idx[d] = bidx%shape[d];
//   //   bidx /= shape[d];
//   // }
//   // for (int i = 1; i < assigned_ny; i++) {
//   //   int d = assigned_dimy[i];
//   //   idx[d] = bidy%shape[d];
//   //   bidy /= shape[d];
//   // }
//   // for (int i = 1; i < assigned_nz; i++) {
//   //   int d = assigned_dimz[i];
//   //   idx[d] = bidz%shape[d];
//   //   bidz /= shape[d];
//   // }
// }

// template <typename T> T max_norm_cuda(const T *v, size_t size);

template <typename T> __device__ T _get_dist(T *coords, int i, int j);

// // __host__ __device__ int get_lindex_cuda(const int n, const int no, const int i);







// template <typename T>
// __device__ inline T tridiag_forward(T prev, T bm, T curr) {

// #ifdef MGARD_CUDA_FMA
//   if (sizeof(T) == sizeof(double)) {
//     return fma(prev, bm, curr);
//   } else if (sizeof(T) == sizeof(float)) {
//     return fmaf(prev, bm, curr);
//   }
// #else
//   return curr - prev * bm;
// #endif
// }

// template <typename T>
// __device__ inline T tridiag_backward(T prev, T dist, T am, T curr) {

// #ifdef MGARD_CUDA_FMA
//   if (sizeof(T) == sizeof(double)) {
//     return fma(-1 * dist, prev, curr) * am;
//   } else if (sizeof(T) == sizeof(float)) {
//     return fmaf(-1 * dist, prev, curr) * am;
//   }
// #else
//   return (curr - dist * prev) / am;
// #endif
// }




} // namespace mgard_cuda

#include "MemoryManagement.hpp"
#include "Array.hpp"
#include "SubArray.hpp"
#include "Metadata.h"
#include "AutoTuners/AutoTuner.h"
#include "Task.h"
#include "Utilities/Timer.hpp"
// #include "Functor.h"
// #include "AutoTuner.h"
// #include "Task.h"
// #include "DeviceAdapters/DeviceAdapter.h"

#endif

