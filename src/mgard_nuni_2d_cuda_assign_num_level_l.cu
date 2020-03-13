#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>
#include <cmath>

namespace mgard_2d {
namespace mgard_gen {

template <typename T>
__global__ void 
_assign_num_level_l_cuda(int nrow,           int ncol,
                         int nr,             int nc,
                         int row_stride,     int col_stride,
                         int * dirow,        int * dicol,
                         T * dv,        int lddv,
                         T num) {
  
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;

  for (int y = y0; y < nr; y += blockDim.y * gridDim.y * row_stride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x * col_stride) {
      dv[get_idx(lddv, dirow[y], dicol[x])] = num;
    }
  }
}

template <typename T>
mgard_cuda_ret 
assign_num_level_l_cuda(int nrow,           int ncol,
                        int nr,             int nc,
                        int row_stride,     int col_stride,
                        int * dirow,        int * dicol,
                        T * dv,        int lddv,
                        T num,
                        int B, 
                        mgard_cuda_handle & handle, 
                        int queue_idx, bool profile) {

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_y = ceil((float)nr/(row_stride));
  int total_thread_x = ceil((float)nc/(col_stride));
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _assign_num_level_l_cuda<<<blockPerGrid, threadsPerBlock,
                             0, stream>>>(nrow,       ncol,
                                          nr,         nc,
                                          row_stride, col_stride,
                                          dirow,      dicol,
                                          dv,         lddv,
                                          num);
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
_assign_num_level_l_cuda_l2_sm(int nr,             int nc,
                               int row_stride,     int col_stride,
                               T * dv,        int lddv,
                               T num) {
  
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;

  for (int y = y0; y < nr; y += blockDim.y * gridDim.y * row_stride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x * col_stride) {
      dv[get_idx(lddv, y, x)] = num;
    }
  }
}

template <typename T>
mgard_cuda_ret 
assign_num_level_l_cuda_l2_sm(int nr,             int nc,
                              int row_stride,     int col_stride,
                              T * dv,        int lddv,
                              T num,
                              int B, mgard_cuda_handle & handle, 
                              int queue_idx, bool profile) {
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_y = ceil((float)nr/(row_stride));
  int total_thread_x = ceil((float)nc/(col_stride));
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _assign_num_level_l_cuda_l2_sm<<<blockPerGrid, threadsPerBlock,
                                   0, stream>>>(nr,         nc,
                                                row_stride, col_stride,
                                                dv,         lddv,
                                                num);
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
assign_num_level_l_cuda<double>(int nrow,           int ncol,
                        int nr,             int nc,
                        int row_stride,     int col_stride,
                        int * dirow,        int * dicol,
                        double * dv,        int lddv,
                        double num,
                        int B, mgard_cuda_handle & handle, 
                        int queue_idx, bool profile);

template mgard_cuda_ret 
assign_num_level_l_cuda<float>(int nrow,           int ncol,
                        int nr,             int nc,
                        int row_stride,     int col_stride,
                        int * dirow,        int * dicol,
                        float * dv,        int lddv,
                        float num,
                        int B, mgard_cuda_handle & handle, 
                        int queue_idx, bool profile);

template mgard_cuda_ret 
assign_num_level_l_cuda_l2_sm<double>(int nr,             int nc,
                                      int row_stride,     int col_stride,
                                      double * dv,        int lddv,
                                      double num,
                                      int B, mgard_cuda_handle & handle, 
                                      int queue_idx, bool profile);
template mgard_cuda_ret 
assign_num_level_l_cuda_l2_sm<float>(int nr,             int nc,
                                     int row_stride,     int col_stride,
                                     float * dv,        int lddv,
                                     float num,
                                     int B, mgard_cuda_handle & handle, 
                                     int queue_idx, bool profile);




}
}