#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_nuni_2d_cuda_kernels.h"
#include <fstream>
#include <cmath>

namespace mgard_2d {
namespace mgard_gen {  

template <typename T>
__global__ void
_prolongate_l_row_cuda(int nrow,       int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       T * dv,    int lddv,
                       T * coords_x) {

  int col_Pstride = col_stride / 2;
  int r0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  for (int r = r0; r < nr; r += (blockDim.x * gridDim.x) * row_stride) {
    T * vec = dv + dirow[r] * lddv;
    for (int i = col_stride; i < nc; i += col_stride) {
      T h1 = mgard_common::_get_dist(coords_x, dicol[i - col_stride], dicol[i - col_Pstride]);
      T h2 = mgard_common::_get_dist(coords_x, dicol[i - col_Pstride], dicol[i]);
      // double h1 = dicol[i - col_Pstride] - dicol[i - col_stride];
      // double h2 = dicol[i] - dicol[i - col_Pstride];
      T hsum = h1 + h2;
      vec[dicol[i - col_Pstride]] = (h2 * vec[dicol[i - col_stride]] + h1 * vec[dicol[i]]) / hsum;
    }
  }
}

template <typename T>
mgard_cuda_ret 
prolongate_l_row_cuda(int nrow,       int ncol,
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      T * dv,    int lddv,
                      T * dcoords_x,
                      int B, mgard_cuda_handle & handle, 
                      int queue_idx, bool profile) {
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread_x = ceil((float)nr/(row_stride));
  //int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  //int gridy = ceil(total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }


  _prolongate_l_row_cuda<<<blockPerGrid, threadsPerBlock,
                           0, stream>>>(nrow,       ncol,
                                        nr,         nc,
                                        row_stride, col_stride,
                                        dirow,      dicol,
                                        dv,         lddv,
                                        dcoords_x);
  gpuErrchk(cudaGetLastError ());

  if (profile) {
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }

  return mgard_cuda_ret(0, milliseconds/1000.0);
}

template <typename T>
__global__ void
_prolongate_l_col_cuda(int nrow,       int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       T * dv,    int lddv,
                       T * coords_y) {

  int row_Pstride = row_stride / 2;
  int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
    T * vec = dv + dicol[c];
    for (int i = row_stride; i < nr; i += row_stride) {
      T h1 = mgard_common::_get_dist(coords_y, dirow[i - row_stride], dirow[i - row_Pstride]);
      T h2 = mgard_common::_get_dist(coords_y, dirow[i - row_Pstride], dirow[i]);
      // double h1 = dirow[i - row_Pstride] - dirow[i - row_stride];
      // double h2 = dirow[i] - dirow[i - row_Pstride];
      T hsum = h1 + h2;
      vec[dirow[i - row_Pstride] * lddv] = (h2 * vec[dirow[i - row_stride] * lddv] + h1 * vec[dirow[i] * lddv]) / hsum;
    }
  }
}

template <typename T>
mgard_cuda_ret 
prolongate_l_col_cuda(int nrow,        int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       T * dv,    int lddv,
                       T * dcoords_y,
                       int B, mgard_cuda_handle & handle, 
                       int queue_idx, bool profile) {
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  //int total_thread_y = ceil((double)nrow/(row_stride));
  int total_thread_x = ceil((double)ncol/(col_stride));
  //int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  //int gridy = ceil(total_thread_y/tby);
  int gridx = ceil((double)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _prolongate_l_col_cuda<<<blockPerGrid, threadsPerBlock,
                           0, stream>>>(nrow,       ncol,
                                        nr,         nc,
                                        row_stride, col_stride,
                                        dirow,      dicol,
                                        dv,         lddv,
                                        dcoords_y);
  gpuErrchk(cudaGetLastError ());

  if (profile) {
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }

  return mgard_cuda_ret(0, milliseconds/1000.0);
}

template <typename T>
__global__ void
_prolongate_l_row_cuda_sm(int nr,         int nc,
                          int row_stride, int col_stride,
                          T * dv,    int lddv,
                          T * ddist_x) {

  // register int row_stride = min(row_row_stride, col_row_stride);
  // register int col_stride = min(row_col_stride, col_col_stride);

  register int row_row_stride_ratio = 1;//row_row_stride / row_stride;
  register int row_col_stride_ratio = 1;//row_col_stride / col_stride;
  // register int col_row_stride_ratio = col_row_stride / row_stride;
  // register int col_col_stride_ratio = col_col_stride / col_stride;



  register int c0 = blockIdx.x * blockDim.x;
  register int c0_stride = c0 * col_stride;
  register int r0 = blockIdx.y * blockDim.y;
  register int r0_stride = r0 * row_stride;

  register int total_row = ceil((double)nr/(row_stride));
  register int total_col = ceil((double)nc/(col_stride));

  register int c_sm = threadIdx.x;
  register int r_sm = threadIdx.y;

  extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  T * sm = reinterpret_cast<T *>(smem);
  //extern __shared__ double sm[]; // size: (blockDim.x + 1) * (blockDim.y + 1)
  int ldsm = blockDim.x + 1;
  T * v_sm = sm;
  T * dist_x_sm = sm + (blockDim.x + 1) * (blockDim.y + 1);
  T * dist_y_sm = dist_x_sm + blockDim.x;

  for (int r = r0; r < total_row - 1; r += blockDim.y * gridDim.y) {
    for (int c = c0; c < total_col - 1; c += blockDim.x * gridDim.x) {
      /* Load v */
      if (c + c_sm < total_col && r + r_sm < total_row) {
        v_sm[r_sm * ldsm + c_sm] = dv[(r + r_sm) * row_stride * lddv + (c + c_sm) * col_stride];

        if (r_sm == 0 && r + blockDim.y < total_row) {
          v_sm[blockDim.y * ldsm + c_sm] = dv[(r + blockDim.y) * row_stride * lddv + (c + c_sm) * col_stride];
        }
        if (c_sm == 0 && c + blockDim.x < total_col) {
          v_sm[r_sm * ldsm + blockDim.x] = dv[(r + r_sm) * row_stride * lddv + (c + blockDim.x) * col_stride];
        }
        if (r_sm == 0 && c_sm == 0 && r + blockDim.y < total_row && c + blockDim.x < total_col) {
          v_sm[blockDim.y * ldsm + blockDim.x] = dv[(r + blockDim.y) * row_stride * lddv + (c + blockDim.x) * col_stride];
        }
      }

      /* Load dist_x */
      if (c + c_sm < total_col) {
        dist_x_sm[c_sm] = ddist_x[c + c_sm];
      }
      // /* Load dist_y */
      // if (r + r_sm < total_row) {
      //   dist_y_sm[r_sm] = ddist_y[r + r_sm];
      //   // printf("load ddist_y[%d] %f\n", r_sm, dist_y_sm[r_sm]);
      // }

      __syncthreads();

      /* Compute row by row*/
      if (r_sm % row_row_stride_ratio == 0 && c_sm % (2*row_col_stride_ratio) != 0) {
        T h1 = dist_x_sm[c_sm - row_col_stride_ratio];
        T h2 = dist_x_sm[c_sm];
        // if (r0 + r_sm == 0) {
        //   printf("h1 %f h2 %f v1 %f v2 %f\n", h1, h2, v_sm[r_sm * ldsm + (c_sm - row_col_stride_ratio)], v_sm[r_sm * ldsm + (c_sm + row_col_stride_ratio)]);
        // }
        v_sm[r_sm * ldsm + c_sm] = (h2 * v_sm[r_sm * ldsm + (c_sm - row_col_stride_ratio)] + 
                                     h1 * v_sm[r_sm * ldsm + (c_sm + row_col_stride_ratio)])/
                                    (h1 + h2);
        dv[(r + r_sm) * row_stride * lddv + (c + c_sm) * col_stride] = v_sm[r_sm * ldsm + c_sm];
      } 
      /* extra computaion for global boarder */
      if (r + blockDim.y == total_row - 1) {
        if (r_sm == 0 && c_sm % (2*row_col_stride_ratio) != 0) {
          T h1 = dist_x_sm[c_sm - row_col_stride_ratio];
          T h2 = dist_x_sm[c_sm];
          v_sm[blockDim.y * ldsm + c_sm] = (h2 * v_sm[blockDim.y * ldsm + (c_sm - row_col_stride_ratio)] + 
                                             h1 * v_sm[blockDim.y * ldsm + (c_sm + row_col_stride_ratio)])/
                                            (h1 + h2);
          dv[(r + blockDim.y) * row_stride * lddv + (c + c_sm) * col_stride] = v_sm[blockDim.y * ldsm + c_sm];
        }
      }

      __syncthreads();
      
    }
  }
}

template <typename T>
mgard_cuda_ret 
prolongate_l_row_cuda_sm(int nr,         int nc,
                         int row_stride, int col_stride,
                         T * dv,    int lddv,
                         T * ddist_x,
                         int B,
                         mgard_cuda_handle & handle, 
                         int queue_idx, bool profile) {
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_row = ceil((double)nr/(row_stride));
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = total_row - 1;
  int total_thread_x = total_col - 1;
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  size_t sm_size = ((B+1) * (B+1) + 2 * B) * sizeof(T);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _prolongate_l_row_cuda_sm<<<blockPerGrid, threadsPerBlock, 
                              sm_size, stream>>>(nr,         nc,
                                                 row_stride, col_stride,
                                                 dv,         lddv,
                                                 ddist_x);
  gpuErrchk(cudaGetLastError ());

  if (profile) {
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }

  return mgard_cuda_ret(0, milliseconds/1000.0);
}

template <typename T>
__global__ void
_prolongate_l_col_cuda_sm(int nr,         int nc,
                          int row_stride, int col_stride,
                          T * dv,    int lddv,
                          T * ddist_y) {

  // register int row_stride = min(row_row_stride, col_row_stride);
  // register int col_stride = min(row_col_stride, col_col_stride);

  // register int row_row_stride_ratio = row_row_stride / row_stride;
  // register int row_col_stride_ratio = row_col_stride / col_stride;
  register int col_row_stride_ratio = 1; //col_row_stride / row_stride;
  register int col_col_stride_ratio = 1; //col_col_stride / col_stride;



  register int c0 = blockIdx.x * blockDim.x;
  register int c0_stride = c0 * col_stride;
  register int r0 = blockIdx.y * blockDim.y;
  register int r0_stride = r0 * row_stride;

  register int total_row = ceil((double)nr/(row_stride));
  register int total_col = ceil((double)nc/(col_stride));

  register int c_sm = threadIdx.x;
  register int r_sm = threadIdx.y;

  // if (r0 + r_sm == 0 && c0 + c_sm == 0) {
  //   printf("row_row_stride = %d\n", row_row_stride);
  //   printf("row_col_stride = %d\n", row_col_stride);
  //   printf("col_row_stride = %d\n", col_row_stride);
  //   printf("col_col_stride = %d\n", col_col_stride);


  //   printf("row_row_stride_ratio = %d\n", row_row_stride_ratio);
  //   printf("row_col_stride_ratio = %d\n", row_col_stride_ratio);
  //   printf("col_row_stride_ratio = %d\n", col_row_stride_ratio);
  //   printf("col_col_stride_ratio = %d\n", col_col_stride_ratio);
  // } 

  extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  T * sm = reinterpret_cast<T *>(smem);
  //extern __shared__ double sm[]; // size: (blockDim.x + 1) * (blockDim.y + 1)
  int ldsm = blockDim.x + 1;
  T * v_sm = sm;
  T * dist_x_sm = sm + (blockDim.x + 1) * (blockDim.y + 1);
  T * dist_y_sm = dist_x_sm + blockDim.x;

  for (int r = r0; r < total_row - 1; r += blockDim.y * gridDim.y) {
    for (int c = c0; c < total_col - 1; c += blockDim.x * gridDim.x) {
      /* Load v */
      if (c + c_sm < total_col && r + r_sm < total_row) {
        v_sm[r_sm * ldsm + c_sm] = dv[(r + r_sm) * row_stride * lddv + (c + c_sm) * col_stride];

        if (r_sm == 0 && r + blockDim.y < total_row) {
          v_sm[blockDim.y * ldsm + c_sm] = dv[(r + blockDim.y) * row_stride * lddv + (c + c_sm) * col_stride];
        }
        if (c_sm == 0 && c + blockDim.x < total_col) {
          v_sm[r_sm * ldsm + blockDim.x] = dv[(r + r_sm) * row_stride * lddv + (c + blockDim.x) * col_stride];
        }
        if (r_sm == 0 && c_sm == 0 && r + blockDim.y < total_row && c + blockDim.x < total_col) {
          v_sm[blockDim.y * ldsm + blockDim.x] = dv[(r + blockDim.y) * row_stride * lddv + (c + blockDim.x) * col_stride];
        }
      }

      /* Load dist_y */
      if (r + r_sm < total_row) {
        dist_y_sm[r_sm] = ddist_y[r + r_sm];
        // printf("load ddist_y[%d] %f\n", r_sm, dist_y_sm[r_sm]);
      }

      __syncthreads();
      
      /* Compute col by col*/
      if (r_sm % (2*col_row_stride_ratio) != 0 && c_sm % col_col_stride_ratio == 0) {
        T h1 = dist_y_sm[r_sm - col_row_stride_ratio];
        T h2 = dist_y_sm[r_sm];
        v_sm[r_sm * ldsm + c_sm] = (h2 * v_sm[(r_sm - col_row_stride_ratio) * ldsm + c_sm] +
                                     h1 * v_sm[(r_sm + col_row_stride_ratio) * ldsm + c_sm])/
                                    (h1 + h2);
        dv[(r + r_sm) * row_stride * lddv + (c + c_sm) * col_stride] = v_sm[r_sm * ldsm + c_sm];
      } 
      /* extra computaion for global boarder */
      if (c + blockDim.x == total_col - 1) {
        if (r_sm % (2*col_row_stride_ratio) != 0 && c_sm == 0) {
          T h1 = dist_y_sm[r_sm - col_row_stride_ratio];
          T h2 = dist_y_sm[r_sm];
          v_sm[r_sm * ldsm + blockDim.x] = (h2 * v_sm[(r_sm - col_row_stride_ratio) * ldsm + blockDim.x] +
                                             h1 * v_sm[(r_sm + col_row_stride_ratio) * ldsm + blockDim.x])/
                                            (h1 + h2);
          dv[(r + r_sm) * row_stride * lddv + (c + blockDim.x) * col_stride] = v_sm[r_sm * ldsm + blockDim.x];
        } 
      }
      
      __syncthreads();
    }
  }
}

template <typename T>
mgard_cuda_ret 
prolongate_l_col_cuda_sm(int nr,         int nc,
                         int row_stride, int col_stride,
                         T * dv,    int lddv,
                         T * ddist_y,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile) {
  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_row = ceil((double)nr/(row_stride));
  int total_col = ceil((double)nc/(col_stride));
  int total_thread_y = total_row - 1;
  int total_thread_x = total_col - 1;
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  size_t sm_size = ((B+1) * (B+1) + 2 * B) * sizeof(T);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  
  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _prolongate_l_col_cuda_sm<<<blockPerGrid, threadsPerBlock, 
                              sm_size, stream>>>(nr,         nc,
                                                row_stride, col_stride,
                                                dv,         lddv,
                                                ddist_y);
  gpuErrchk(cudaGetLastError ());

  if (profile) {
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
  }
  return mgard_cuda_ret(0, milliseconds/1000.0);
}

template mgard_cuda_ret 
prolongate_l_row_cuda<double>(int nrow,       int ncol,
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      double * dv,    int lddv,
                      double * dcoords_x,
                      int B, mgard_cuda_handle & handle, 
                      int queue_idx, bool profile);
template mgard_cuda_ret 
prolongate_l_row_cuda<float>(int nrow,       int ncol,
                      int nr,         int nc,
                      int row_stride, int col_stride,
                      int * dirow,    int * dicol,
                      float * dv,    int lddv,
                      float * dcoords_x,
                      int B, mgard_cuda_handle & handle, 
                      int queue_idx, bool profile);

template mgard_cuda_ret 
prolongate_l_col_cuda<double>(int nrow,        int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       double * dv,    int lddv,
                       double * dcoords_y, 
                       int B, mgard_cuda_handle & handle, 
                       int queue_idx, bool profile);
template mgard_cuda_ret 
prolongate_l_col_cuda<float>(int nrow,        int ncol,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       int * dirow,    int * dicol,
                       float * dv,    int lddv,
                       float * dcoords_y,
                       int B, mgard_cuda_handle & handle, 
                       int queue_idx, bool profile);

template  mgard_cuda_ret 
prolongate_l_row_cuda_sm<double>(int nr,         int nc,
                                 int row_stride, int col_stride,
                                 double * dv,    int lddv,
                                 double * ddist_x,
                                 int B, mgard_cuda_handle & handle, 
                                 int queue_idx, bool profile);
template  mgard_cuda_ret 
prolongate_l_row_cuda_sm<float>(int nr,         int nc,
                                 int row_stride, int col_stride,
                                 float * dv,    int lddv,
                                 float * ddist_x,
                                 int B, mgard_cuda_handle & handle, 
                                 int queue_idx, bool profile);

template mgard_cuda_ret 
prolongate_l_col_cuda_sm<double>(int nr,         int nc,
                                 int row_stride, int col_stride,
                                 double * dv,    int lddv,
                                 double * ddist_y,
                                 int B, mgard_cuda_handle & handle, 
                                 int queue_idx, bool profile);
template mgard_cuda_ret 
prolongate_l_col_cuda_sm<float>(int nr,         int nc,
                                 int row_stride, int col_stride,
                                 float * dv,    int lddv,
                                 float * ddist_y,
                                 int B, mgard_cuda_handle & handle, 
                                 int queue_idx, bool profile);


}
}
