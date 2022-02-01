/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "cuda/CommonInternal.h"
#include "cuda/PrecomputeKernels.h"
#include <iomanip>
#include <iostream>

namespace mgard_x {

template <typename T>
__global__ void _calc_cpt_dist(int n, T *dcoord, T *ddist) {

  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);
  T *sm = SharedMemory<T>();
  // extern __shared__ double sm[]; //size = blockDim.x + 1

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int x0_sm = threadIdx.x;
  T dist;
  for (int x = x0; x < n; x += blockDim.x * gridDim.x) {
    // Load coordinates
    sm[x0_sm] = dcoord[x];
    // printf("sm[%d] block %d thread %d load[%d] %f\n", x0_sm, blockIdx.x,
    // threadIdx.x, x, dcoord[x * stride]);
    if (x0_sm == 0) {
      // sm[blockDim.x] = dcoord[(x + blockDim.x) * stride];
      int left = n - blockIdx.x * blockDim.x;
      if (left >= blockDim.x + 1) {
        sm[blockDim.x] = dcoord[blockDim.x + x];
      }
      // sm[min(blockDim.x, left - 1)] =
      //     dcoord[min((x + blockDim.x) * stride, n - 1)];
      // printf("sm[%d] extra block %d thread %d load[%d] %f\n", min(blockDim.x,
      // left-1), blockIdx.x, threadIdx.x, min((x + blockDim.x) * stride, n-1),
      // dcoord[min((x + blockDim.x) * stride, n-1)]);
    }
    __syncthreads();

    // Compute distance

    if (x < n - 1) {
      ddist[x] = _get_dist(sm, x0_sm, x0_sm + 1);
    }
    __syncthreads();
    // ddist[x] = dist;
    // __syncthreads();
  }
}

template <DIM D, typename T>
void calc_cpt_dist(Handle<D, T> &handle, int n, T *dcoord, T *ddist,
                   int queue_idx) {

  int total_thread_x = std::max(n, 1);
  int total_thread_y = 1;
  int tbx = std::min(16, total_thread_x);
  int tby = 1;
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  size_t sm_size = (tbx + 1) * sizeof(T);

  // printf("sm %d (%d %d) (%d %d)\n", sm_size, tbx, tby, gridx, gridy);
  _calc_cpt_dist<<<blockPerGrid, threadsPerBlock, sm_size,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(n, dcoord, ddist);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_X_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template <typename T>
__global__ void _reduce_two_dist(int n, T *ddist, T *ddist_reduced) {
  int x_gl = blockIdx.x * blockDim.x + threadIdx.x;
  int x_sm = threadIdx.x;
  T *sm = SharedMemory<T>();
  if (x_gl < n) {
    sm[x_sm] = ddist[x_gl];
    // printf("thread %d load %f\n", x_gl, ddist[x_gl]);
    __syncthreads();
    if (x_gl % 2 == 0) {
      ddist_reduced[x_gl / 2] = sm[x_sm] + sm[x_sm + 1];
      // printf("thread %d compute %f + %f -> [%d]%f\n", x_gl, sm[x_sm],
      // sm[x_sm+1], x_gl / 2, ddist_reduced[x_gl / 2]);
    }
    // __syncthreads();
    // if (x_gl % 2 == 0) {
    //   dratio[x_gl / 2] = ddist_reduced[x_gl / 2] / (ddist_reduced[x_gl / 2] +
    //   ddist_reduced[x_gl / 2 + 1]);
    // }
  }
}

template <DIM D, typename T>
void reduce_two_dist(Handle<D, T> &handle, int n, T *ddist, T *ddist_reduced,
                     int queue_idx) {

  int total_thread_x = std::max(n, 1);
  int total_thread_y = 1;
  int tbx = std::min(16, total_thread_x);
  int tby = 1;
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  size_t sm_size = tbx * sizeof(T);
  // printf("reduce_two_dist: n: %d\n", n);
  // printf("sm %d (%d %d) (%d %d)\n", sm_size, tbx, tby, gridx, gridy);
  _reduce_two_dist<<<blockPerGrid, threadsPerBlock, sm_size,
                     *(cudaStream_t *)handle.get(queue_idx)>>>(n, ddist,
                                                               ddist_reduced);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_X_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template <typename T>
__global__ void _dist_to_ratio(int n, T *ddist, T *dratio) {

  T *sm = SharedMemory<T>();

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int x_sm = threadIdx.x;
  if (x < n) {
    // Load dists
    sm[x_sm] = ddist[x];
    if (x_sm == 0) {
      int left = n - blockIdx.x * blockDim.x;
      if (left >= blockDim.x + 1) {
        sm[blockDim.x] = ddist[blockDim.x + x];
      }
      // if (blockIdx.x == 1) {
      //   for (int i = 0; i < blockDim.x + 1; i++) {
      //     printf("%f ", sm[i]);
      //   }
      //   printf("\n");
      // }
    }
  }
  __syncthreads();
  // Compute distance
  if (x < n - 1) {
    dratio[x] = sm[x_sm] / (sm[x_sm] + sm[x_sm + 1]);
    // if (blockIdx.x == 1) {
    // printf("x(%d) %f %f %f\n", x, dratio[x], sm[x_sm],sm[x_sm+1]);}
  }
}

template <DIM D, typename T>
void dist_to_ratio(Handle<D, T> &handle, int n, T *ddist, T *dratio,
                   int queue_idx) {

  int total_thread_x = std::max(n, 1);
  int total_thread_y = 1;
  int tbx = std::min(16, total_thread_x);
  int tby = 1;
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  size_t sm_size = (tbx + 1) * sizeof(T);
  // printf("reduce_two_dist: n: %d\n", n);
  // printf("sm %d (%d %d) (%d %d)\n", sm_size, tbx, tby, gridx, gridy);
  _dist_to_ratio<<<blockPerGrid, threadsPerBlock, sm_size,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(n, ddist, dratio);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_X_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template <typename T>
__global__ void _dist_to_volume(int n, T *ddist, T *dvolume) {
  T *sm = SharedMemory<T>();
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int x_sm = threadIdx.x;
  if (x < n - 1) {
    // Load dist
    sm[x_sm + 1] = ddist[x];
  }
  if (x == n - 1) {
    sm[x_sm + 1] = 0;
  }
  int left = n - blockIdx.x * blockDim.x;
  if (x_sm == 0) { // load extra
    sm[0] = x - 1 < 0 ? 0 : ddist[x - 1];
  }

  // if (threadIdx.x == 0) {
  //   for (int i = 0; i < blockDim.x+2; i++) {
  // printf("dist[%d] = %f\n", i, sm[i]);
  //   }
  // }
  int node_coeff_div = n / 2 + 1;
  if (n == 2) {
    dvolume[x] = (sm[x_sm] + sm[x_sm + 1]) / 2;
  } else {
    if (n % 2 != 0) {
      if (x % 2 == 0) { // node
        dvolume[x / 2] = (sm[x_sm] + sm[x_sm + 1]) / 2;
      } else { // coeff
        dvolume[node_coeff_div + x / 2] = (sm[x_sm] + sm[x_sm + 1]) / 2;
      }
    } else {
      if (x != n - 1) {
        if (x % 2 == 0) { // node
          dvolume[x / 2] = (sm[x_sm] + sm[x_sm + 1]) / 2;
          // printf("%f <- %f %f\n", dvolume[x/2], sm[x_sm], sm[x_sm+1]);
        } else { // coeff
          dvolume[node_coeff_div + x / 2] = (sm[x_sm] + sm[x_sm + 1]) / 2;
          // printf("%f <- %f %f\n", dvolume[node_coeff_div + x/2], sm[x_sm],
          // sm[x_sm+1]);
        }
      } else {
        dvolume[x / 2 + 1] = (sm[x_sm] + sm[x_sm + 1]) / 2;
        // printf("%f <- %f %f\n", dvolume[x/2+1], sm[x_sm], sm[x_sm+1]);
      }
    }
  }
}

template <DIM D, typename T>
void dist_to_volume(Handle<D, T> &handle, int n, T *ddist, T *dvolume,
                    int queue_idx) {

  int total_thread_x = std::max(n, 1);
  int total_thread_y = 1;
  int tbx = std::min(16, total_thread_x);
  int tby = 1;
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  size_t sm_size = (tbx + 2) * sizeof(T);
  // printf("reduce_two_dist: n: %d\n", n);
  // printf("sm %d (%d %d) (%d %d)\n", sm_size, tbx, tby, gridx, gridy);
  _dist_to_volume<<<blockPerGrid, threadsPerBlock, sm_size,
                    *(cudaStream_t *)handle.get(queue_idx)>>>(n, ddist,
                                                              dvolume);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_X_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template <typename T>
__global__ void _calc_am_bm(int n, T *ddist, T *am, T *bm) {
  int c = threadIdx.x;
  int c_sm = threadIdx.x;
  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);
  T *sm = SharedMemory<T>();
  // extern __shared__ double sm[];
  T *ddist_sm = sm;
  T *am_sm = sm + blockDim.x;
  T *bm_sm = am_sm + blockDim.x;

  T prev_am = 1.0;
  T prev_dist = 0.0;
  int rest = n;

  while (rest > blockDim.x) {
    /* Load ddsist */
    ddist_sm[c_sm] = ddist[c];
    __syncthreads();
    /* Calculation on one thread*/
    if (c_sm == 0) {
      bm_sm[0] = prev_dist / prev_am;
      am_sm[0] = 2.0 * (ddist_sm[0] + prev_dist) - bm_sm[0] * prev_dist;
      for (int i = 1; i < blockDim.x; i++) {
        bm_sm[i] = ddist_sm[i - 1] / am_sm[i - 1];
        am_sm[i] =
            2.0 * (ddist_sm[i - 1] + ddist_sm[i]) - bm_sm[i] * ddist_sm[i - 1];
      }
      prev_am = am_sm[blockDim.x - 1];
      prev_dist = ddist_sm[blockDim.x - 1];
    }
    __syncthreads();
#ifdef MGARD_X_FMA
    am[c] = 1 / am_sm[c_sm];
    bm[c] = bm_sm[c_sm] * -1;
#else
    am[c] = am_sm[c_sm];
    bm[c] = bm_sm[c_sm];
#endif
    __syncthreads();
    c += blockDim.x;
    rest -= blockDim.x;
    __syncthreads();
  } // end of while

  if (c_sm < rest - 1) {
    ddist_sm[c_sm] = ddist[c];
  }

  __syncthreads();
  if (c_sm == 0) {
    if (rest == 1) {
      bm_sm[rest - 1] = prev_dist / prev_am;
      am_sm[rest - 1] = 2.0 * prev_dist - bm_sm[rest - 1] * prev_dist;
      // printf("bm = %f\n", bm_sm[rest-1]);
      // printf("am = %f\n", am_sm[rest-1]);
    } else {
      bm_sm[0] = prev_dist / prev_am;
      am_sm[0] = 2.0 * (ddist_sm[0] + prev_dist) - bm_sm[0] * prev_dist;
      for (int i = 1; i < rest - 1; i++) {
        bm_sm[i] = ddist_sm[i - 1] / am_sm[i - 1];
        am_sm[i] =
            2.0 * (ddist_sm[i - 1] + ddist_sm[i]) - bm_sm[i] * ddist_sm[i - 1];
      }
      bm_sm[rest - 1] = ddist_sm[rest - 2] / am_sm[rest - 2];
      am_sm[rest - 1] =
          2.0 * ddist_sm[rest - 2] - bm_sm[rest - 1] * ddist_sm[rest - 2];
    }
  }
  __syncthreads();
  if (c_sm < rest) {
#ifdef MGARD_X_FMA
    am[c] = 1 / am_sm[c_sm];
    bm[c] = bm_sm[c_sm] * -1;
#else
    am[c] = am_sm[c_sm];
    bm[c] = bm_sm[c_sm];
#endif
  }
}

template <DIM D, typename T>
void calc_am_bm(Handle<D, T> &handle, int n, T *ddist, T *am, T *bm,
                int queue_idx) {

  // int total_thread_y = 1;
  int total_thread_x = 16;
  int tby = 1;
  int tbx = std::min(16, total_thread_x);
  size_t sm_size = 16 * 3 * sizeof(T);
  int gridy = 1;
  int gridx = 1;
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _calc_am_bm<<<blockPerGrid, threadsPerBlock, sm_size,
                *(cudaStream_t *)handle.get(queue_idx)>>>(n, ddist, am, bm);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_X_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

#define KERNELS(D, T)                                                          \
  template void calc_cpt_dist(Handle<D, T> &handle, int n, T *dcoord,          \
                              T *ddist, int queue_idx);                        \
  template void reduce_two_dist<D, T>(Handle<D, T> & handle, int n, T *ddist,  \
                                      T *ddist_reduced, int queue_idx);        \
  template void dist_to_ratio<D, T>(Handle<D, T> & handle, int n, T *ddist,    \
                                    T *dratio, int queue_idx);                 \
  template void dist_to_volume<D, T>(Handle<D, T> & handle, int n, T *ddist,   \
                                     T *dvolume, int queue_idx);               \
  template void calc_am_bm<D, T>(Handle<D, T> & handle, int n, T *ddist,       \
                                 T *am, T *bm, int queue_idx);

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)
#undef KERNELS

} // namespace mgard_x