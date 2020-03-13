#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda_common.h"
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
_restriction_first_row_cuda(int nrow,       int ncol,
                            int row_stride, int * icolP, int nc,
                            T * dv,    int lddv,
                            T * dcoords_x) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  //int y = threadIdx.y * stride;
  for (int x = idx; x < nrow; x += (blockDim.x * gridDim.x) * row_stride) {
    //printf("thread working on %d \n", x);
    T * vec = dv + x * lddv;
    for (int i = 0; i < ncol-nc; i++) {
      T h1 = mgard_common::get_h_cuda(dcoords_x, icolP[i] - 1, 1);
      T h2 = mgard_common::get_h_cuda(dcoords_x, icolP[i]    , 1);
      T hsum = h1 + h2;
      vec[icolP[i] - 1] += h2 * vec[icolP[i]] / hsum;
      vec[icolP[i] + 1] += h1 * vec[icolP[i]] / hsum;
    }

  }
}

template <typename T>
mgard_cuda_ret 
restriction_first_row_cuda(int nrow,       int ncol, 
                           int row_stride, int * dicolP, int nc,
                           T * dv,    int lddv,
                           T * dcoords_x,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile) {
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread = ceil((float)nrow/row_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _restriction_first_row_cuda<<<blockPerGrid, threadsPerBlock,
                                0, stream>>>(nrow,       ncol, 
                                             row_stride, dicolP, nc,
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
_restriction_first_col_cuda(int nrow,       int ncol,
                            int * irowP,    int nr, int col_stride,
                            T * dv,    int lddv,
                            T * dcoords_y) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  //int y = threadIdx.y * stride;
  for (int x = idx; x < ncol; x += (blockDim.x * gridDim.x) * col_stride) {
    //printf("thread working on %d \n", x);
    T * vec = dv + x;
    for (int i = 0; i < nrow-nr; i++) {
      T h1 = mgard_common::get_h_cuda(dcoords_y, irowP[i] - 1, 1);
      T h2 = mgard_common::get_h_cuda(dcoords_y, irowP[i]    , 1);
      T hsum = h1 + h2;
      vec[(irowP[i] - 1) * lddv] += h2 * vec[irowP[i] * lddv] / hsum;
      vec[(irowP[i] + 1) * lddv] += h1 * vec[irowP[i] * lddv] / hsum;
    }
  }
}

template <typename T>
mgard_cuda_ret 
restriction_first_col_cuda(int nrow,       int ncol, 
                           int * dirowP, int nr, int col_stride,
                           T * dv,    int lddv,
                           T * dcoords_y,
                           int B, mgard_cuda_handle & handle,
                            int queue_idx, bool profile) {
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread = ceil((float)ncol/col_stride);
  int tb = min(B, total_thread);
  int grid = ceil((float)total_thread/tb);
  dim3 threadsPerBlock(tb, 1);
  dim3 blockPerGrid(grid, 1);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _restriction_first_col_cuda<<<blockPerGrid, threadsPerBlock,
                                0, stream>>>(nrow,       ncol, 
                                             dirowP, nr, col_stride,
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

template mgard_cuda_ret 
restriction_first_row_cuda<double>(int nrow,       int ncol, 
                           int row_stride, int * dicolP, int nc,
                           double * dv,    int lddv,
                           double * dcoords_x,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile);
template mgard_cuda_ret 
restriction_first_row_cuda<float>(int nrow,       int ncol, 
                           int row_stride, int * dicolP, int nc,
                           float * dv,    int lddv,
                           float * dcoords_x,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile);
template mgard_cuda_ret 
restriction_first_col_cuda<double>(int nrow,       int ncol, 
                           int * dirowP, int nr, int col_stride,
                           double * dv,    int lddv,
                           double * dcoords_y,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile);
template mgard_cuda_ret 
restriction_first_col_cuda<float>(int nrow,       int ncol, 
                           int * dirowP, int nr, int col_stride,
                           float * dv,    int lddv,
                           float * dcoords_y,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile);



}
}