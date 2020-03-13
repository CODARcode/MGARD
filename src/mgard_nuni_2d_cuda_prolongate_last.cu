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
_prolongate_last_row_cuda(int nrow,       int ncol,
                          int nr,         int nc,
                          int row_stride, int col_stride,
                          int * dirow,    int * dicolP,
                          T * dv,    int lddv,
                          T * dcoords_x) {
  int r0 = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  //int y = threadIdx.y * stride;
  for (int r = r0; r < nr; r += (blockDim.x * gridDim.x) * row_stride) {
    // printf("thread %d working on row %d \n", r0, dirow[r]);
    T * vec = dv + dirow[r] * lddv;
    for (int i = 0; i < ncol-nc; i++) {
      T h1 = 1;//mgard_common::get_h_cuda(dcoords_x, icolP[i] - 1, 1);
      T h2 = 1;//mgard_common::get_h_cuda(dcoords_x, icolP[i]    , 1);
      T hsum = h1 + h2;
      // printf("thread %d working on vec = %f %f %f \n", r0, vec[dicolP[i] - 1], vec[dicolP[i]], vec[dicolP[i] + 1]);
      vec[dicolP[i]] = (h2 * vec[dicolP[i] - 1] + h1 * vec[dicolP[i] + 1]) / hsum;
      // printf("thread %d working on vec = %f \n", r0, vec[dicolP[i]]);
    }

  }
}

template <typename T>
mgard_cuda_ret 
prolongate_last_row_cuda(int nrow,       int ncol, 
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * dirow,    int * dicolP,
                         T * dv,    int lddv,
                         T * dcoords_x,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile) {

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_x = ceil((float)nr/row_stride);
  int tbx = min(B, total_thread_x);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _prolongate_last_row_cuda<<<blockPerGrid, threadsPerBlock, 
                              0, stream>>>(nrow,       ncol,
                                           nr,         nc, 
                                           row_stride, col_stride,
                                           dirow,      dicolP,
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
_prolongate_last_col_cuda(int nrow,       int ncol,
                          int nr,         int nc,
                          int row_stride, int col_stride,
                          int * dirowP,   int * dicol, 
                          T * dv,    int lddv,
                          T * dcoords_y) {
  int c0 = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  //int y = threadIdx.y * stride;
  //for (int c = c0; c < nc; c += (blockDim.x * gridDim.x) * col_stride) {
  for (int c = c0; c < ncol; c += (blockDim.x * gridDim.x) * col_stride) {
    //printf("thread working on %d \n", x);
    //double * vec = dv + dicol[c];
    T * vec = dv + c;
    for (int i = 0; i < nrow-nr; i++) {
      T h1 = 1; //mgard_common::get_h_cuda(dcoords_y, irowP[i] - 1, 1);
      T h2 = 1; //mgard_common::get_h_cuda(dcoords_y, irowP[i]    , 1);
      T hsum = h1 + h2;
      // printf("thread %d working on vec = %f %f %f \n", c0, vec[(dirowP[i] - 1)*lddv], vec[dirowP[i]*lddv], vec[(dirowP[i] + 1)*lddv]);
      vec[dirowP[i] * lddv] = (h2 * vec[(dirowP[i] - 1) * lddv] + h1 * vec[(dirowP[i] + 1) * lddv]) / hsum;
      // printf("thread %d working on vec = %f \n", c0, vec[dirowP[i] * lddv]);
    }
  }
}

template <typename T>
mgard_cuda_ret 
prolongate_last_col_cuda(int nrow,       int ncol,
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * dirowP,   int * dicol, 
                         T * dv,    int lddv,
                         T * dcoords_y,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile) {
  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  //int total_thread_x = ceil((float)nc/col_stride);
  int total_thread_x = ceil((float)ncol/col_stride);
  int tbx = min(B, total_thread_x);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, 1);
  dim3 blockPerGrid(gridx, 1);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _prolongate_last_col_cuda<<<blockPerGrid, threadsPerBlock,
                              0, stream>>>(nrow,       ncol, 
                                           nr,         nc,
                                           row_stride, col_stride,
                                           dirowP,     dicol,
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
prolongate_last_row_cuda<double>(int nrow,       int ncol, 
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * dirow,    int * dicolP,
                         double * dv,    int lddv,
                         double * dcoords_x,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);
template mgard_cuda_ret 
prolongate_last_row_cuda<float>(int nrow,       int ncol, 
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * dirow,    int * dicolP,
                         float * dv,    int lddv,
                         float * dcoords_x, 
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);

template mgard_cuda_ret 
prolongate_last_col_cuda<double>(int nrow,       int ncol,
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * dirowP,   int * dicol, 
                         double * dv,    int lddv,
                         double * dcoords_y,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);
template mgard_cuda_ret 
prolongate_last_col_cuda<float>(int nrow,       int ncol,
                         int nr,         int nc,
                         int row_stride, int col_stride,
                         int * dirowP,   int * dicol, 
                         float * dv,    int lddv,
                         float * dcoords_y,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);

}
}