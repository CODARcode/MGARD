#include "cuda/mgard_cuda_common.h"
#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_precompute_kernels.h"
#include <iomanip>
#include <iostream>

namespace mgard_cuda {

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
        sm[blockDim.x] = dcoord[blockDim.x+x];
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

template <typename T, int D>
void calc_cpt_dist(mgard_cuda_handle<T, D> &handle, int n,
                   T *dcoord, T *ddist, int queue_idx) {

  int total_thread_x = std::max(n, 1);
  int total_thread_y = 1;
  int tbx = std::min(16, total_thread_x);
  int tby = 1;
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  size_t sm_size = (tbx + 1) * sizeof(T);

  //printf("sm %d (%d %d) (%d %d)\n", sm_size, tbx, tby, gridx, gridy);
  _calc_cpt_dist<<<blockPerGrid, threadsPerBlock, sm_size,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(n,
                                                             dcoord, ddist);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}


template <typename T>
__global__ void _reduce_two_dist(int n, T *ddist, T *ddist_reduced){
  int x_gl = blockIdx.x * blockDim.x + threadIdx.x;
  int x_sm = threadIdx.x;
  T *sm = SharedMemory<T>();
  if (x_gl < n) {
    sm[x_sm] = ddist[x_gl];
    // printf("thread %d load %f\n", x_gl, ddist[x_gl]);
    __syncthreads();
    if (x_gl % 2 == 0) {
      ddist_reduced[x_gl / 2] = sm[x_sm] + sm[x_sm + 1];
      // printf("thread %d compute %f + %f -> [%d]%f\n", x_gl, sm[x_sm], sm[x_sm+1], x_gl / 2, ddist_reduced[x_gl / 2]);
    }
    // __syncthreads();
    // if (x_gl % 2 == 0) {
    //   dratio[x_gl / 2] = ddist_reduced[x_gl / 2] / (ddist_reduced[x_gl / 2] + ddist_reduced[x_gl / 2 + 1]);
    // }
  }
}


template <typename T, int D>
void reduce_two_dist(mgard_cuda_handle<T, D> &handle, int n, T *ddist, T *ddist_reduced, int queue_idx) {

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
                   *(cudaStream_t *)handle.get(queue_idx)>>>(n, ddist, ddist_reduced);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}



template <typename T>
__global__ void _dist_to_ratio(int n, T *ddist, T *dratio){

  T *sm = SharedMemory<T>();

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int x_sm = threadIdx.x;
  if (x < n) {
    // Load coordinates
    sm[x_sm] = ddist[x];
    if (x_sm == 0) {
      int left = n - blockIdx.x * blockDim.x;
      if (left >= blockDim.x + 1) {
        sm[blockDim.x] = ddist[blockDim.x+x];
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
    dratio[x] = sm[x_sm] / (sm[x_sm] + sm[x_sm+1]);
    // if (blockIdx.x == 1) {
    // printf("x(%d) %f %f %f\n", x, dratio[x], sm[x_sm],sm[x_sm+1]);}
  }
}


template <typename T, int D>
void dist_to_ratio(mgard_cuda_handle<T, D> &handle, int n, T *ddist, T *dratio, int queue_idx) {

  int total_thread_x = std::max(n, 1);
  int total_thread_y = 1;
  int tbx = std::min(16, total_thread_x);
  int tby = 1;
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  size_t sm_size = (tbx+1) * sizeof(T);
  // printf("reduce_two_dist: n: %d\n", n);
  // printf("sm %d (%d %d) (%d %d)\n", sm_size, tbx, tby, gridx, gridy);
  _dist_to_ratio<<<blockPerGrid, threadsPerBlock, sm_size,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(n, ddist, dratio);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}


template <typename T>
__global__ void _calc_cpt_dist_ratio(int n, T *dcoord, T *dratio) {

  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);
  T *sm = SharedMemory<T>();
  // extern __shared__ double sm[]; //size = blockDim.x + 1

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int x0_sm = threadIdx.x;
  T dist1, dist2;

  for (int x = x0; x < n; x += blockDim.x * gridDim.x) {
    // Load coordinates
    sm[x0_sm] = dcoord[x];
    // printf("sm[%d] block %d thread %d load[%d] %f\n", x0_sm, blockIdx.x,
    // threadIdx.x, x, dcoord[x * stride]);
    if (x0_sm == 0) {
      // sm[blockDim.x] = dcoord[(x + blockDim.x) * stride];
      int left = n - blockIdx.x * blockDim.x;
      if (left >= blockDim.x + 2) {
        sm[blockDim.x] = dcoord[x + blockDim.x];
        sm[blockDim.x+1] = dcoord[x + blockDim.x+1];
      } 
      // else {
      // sm[min(blockDim.x, left)] =
      //     dcoord[min((x + blockDim.x) * stride, n - 1)];
      // printf("sm[%d] extra block %d thread %d load[%d] %f\n", min(blockDim.x,
      // left-1), blockIdx.x, threadIdx.x, min((x + blockDim.x) * stride, n-1),
      // dcoord[min((x + blockDim.x) * stride, n-1)]);
      // printf("blockIdx.x: %d left: %d\n", threadIdx.x, left);
      // if (blockIdx.x == 0) {
      //   for (int i = 0; i < blockDim.x + 1; i++) {
      //     printf("%f ", sm[i]);
      //   }
      //   printf("\n");
      // }
    }
    __syncthreads();

    // Compute distance
    if (x < n - 2) {
      dist1 = _get_dist(sm, x0_sm,     x0_sm + 1);
      dist2 = _get_dist(sm, x0_sm + 1, x0_sm + 2);
      dratio[x] = dist1 / (dist1 + dist2); 
      if (blockIdx.x == 0) {
      // printf("ratio(%d) %f = %f / (%f+%f)\n", x, dratio[x], dist1, dist1, dist2);
      }
    }
    __syncthreads();
  }
}

template <typename T, int D>
void calc_cpt_dist_ratio(mgard_cuda_handle<T, D> &handle, int n,
                   T *dcoord, T *dratio, int queue_idx) {

  int total_thread_x = std::max(n, 1);
  int total_thread_y = 1;
  int tbx = std::min(16, total_thread_x);

  int tby = 1;
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  // printf("tbx: %d gridx: %d\n", tbx, gridx);
  size_t sm_size = (tbx + 2) * sizeof(T);
  _calc_cpt_dist_ratio<<<blockPerGrid, threadsPerBlock, sm_size,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(n,
                                                             dcoord, dratio);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
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
#ifdef MGARD_CUDA_FMA
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
#ifdef MGARD_CUDA_FMA
    am[c] = 1 / am_sm[c_sm];
    bm[c] = bm_sm[c_sm] * -1;
#else
    am[c] = am_sm[c_sm];
    bm[c] = bm_sm[c_sm];
#endif
    

  }
}

template <typename T, int D>
void calc_am_bm(mgard_cuda_handle<T, D> &handle, int n, T *ddist, T *am, T *bm,
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
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}


#define KERNELS(T, D) \
                template void calc_cpt_dist(\
                mgard_cuda_handle<T, D> &handle,\
                int n, T *dcoord, T *ddist, int queue_idx);\
                template void reduce_two_dist<T, D>(\
                mgard_cuda_handle<T, D> &handle,\
                int n, T *ddist, T *ddist_reduced, int queue_idx);\
                template void calc_cpt_dist_ratio<T, D>(\
                mgard_cuda_handle<T, D> &handle,\
                int nrow, T *dcoord, T *dratio, int queue_idx);\
                template void dist_to_ratio<T, D>(mgard_cuda_handle<T, D> &handle,\
                int n, T *ddist, T *dratio, int queue_idx);\
                template void calc_am_bm<T, D>(\
                mgard_cuda_handle<T, D> &handle, int n,\
                T *ddist, T *am, T *bm, int queue_idx);
      
KERNELS(double, 1)
KERNELS(float,  1)
KERNELS(double, 2)
KERNELS(float,  2)
KERNELS(double, 3)
KERNELS(float,  3)
KERNELS(double, 4)
KERNELS(float,  4)
KERNELS(double, 5)
KERNELS(float,  5)
#undef KERNELS

}//end namespace