// #include "cuda/mgard_nuni_2d_cuda_common.h"
// #include "cuda/mgard_nuni_2d_cuda_gen.h"
// #include "cuda/mgard_nuni_3d_cuda_common.h"
#include "cuda/mgard_cuda_compact_helper.h"
#include "cuda/mgard_cuda_helper.h"

#include <cstdio>

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

bool is_2kplus1_cuda(double num);

__device__ int get_idx(const int ld, const int i, const int j);

__device__ int get_idx(const int ld1, const int ld2, const int i, const int j,
                       const int k);

template <typename T> T max_norm_cuda(const T *v, size_t size);

template <typename T> __device__ T _get_dist(T *coords, int i, int j);

__host__ __device__ int get_lindex_cuda(const int n, const int no, const int i);

#ifndef MGRAD_CUDA_SHARED_MEMORY
#define MGRAD_CUDA_SHARED_MEMORY

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

#endif
} // namespace mgard_cuda