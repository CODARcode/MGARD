#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"
#include "mgard_nuni_2d_cuda_common.h"
#include <fstream>

namespace mgard_2d {
namespace mgard_cannon {

template <typename T>
__global__ void 
_mass_matrix_multiply_row_cuda(int nrow,       int ncol, 
                               int row_stride, int col_stride,
                               T * dv,    int lddv,
                               T * dcoords_x) {
  //int stride = pow (2, l); // current stride

  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * row_stride;
  //int y = threadIdx.y * stride;
  for (int x = idx; x < nrow; x += (blockDim.x * gridDim.x) * row_stride) {
    //printf("thread working on %d \n", x);
    T * vec = dv + x * lddv;
    register T temp1, temp2;
    temp1 = vec[0];
    // printf("thread %d working on %f\n", idx, temp1);
    vec[0] = 2.0 * mgard_common::get_h_cuda(dcoords_x, 0, col_stride) * temp1 + 
                   mgard_common::get_h_cuda(dcoords_x, 0, col_stride) * vec[col_stride];
    for (int i = col_stride; i < ncol - col_stride; i += col_stride) {
        temp2 = vec[i];
        vec[i] = mgard_common::get_h_cuda(dcoords_x, i - col_stride, col_stride) * temp1 + 
                 2 * 
                    (mgard_common::get_h_cuda(dcoords_x, i - col_stride, col_stride) +
                     mgard_common::get_h_cuda(dcoords_x, i,              col_stride)) *
                 temp2 + 
                 mgard_common::get_h_cuda(dcoords_x, i, col_stride) * vec[i + col_stride];
        temp1 = temp2;
    }
    vec[ncol-1] = mgard_common::get_h_cuda(dcoords_x, ncol - col_stride - 1, col_stride) * temp1 +
              2 * mgard_common::get_h_cuda(dcoords_x, ncol - col_stride - 1, col_stride) * vec[ncol-1];
  }
}

template <typename T>
mgard_cuda_ret 
mass_matrix_multiply_row_cuda(int nrow,       int ncol, 
                              int row_stride, int col_stride,
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

  _mass_matrix_multiply_row_cuda<<<blockPerGrid, threadsPerBlock,
                                   0, stream>>>(nrow,       ncol, 
                                                row_stride, col_stride,
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
_mass_matrix_multiply_col_cuda(int nrow,       int ncol, 
                              int row_stride, int col_stride,
                              T * dv,    int lddv,
                              T * dcoords_y) {
  //int stride = pow (2, l); // current stride

  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * col_stride;
  //int y = threadIdx.y * stride;
  for (int x = idx; x < ncol; x += (blockDim.x * gridDim.x) * col_stride) {
    //printf("thread working on %d \n", x);
    T * vec = dv + x;
    register T temp1, temp2;
    temp1 = vec[0];
    // printf("%d\n", threadIdx.x + blockIdx.x * blockDim.x);
    // if (threadIdx.x + blockIdx.x * blockDim.x == 1)
    //   printf("thread %d x %d working on %f %f %f\n", idx, x, vec[0 * lddv], vec[1 * lddv], vec[2 * lddv]);
    vec[0] = 2.0 * mgard_common::get_h_cuda(dcoords_y, 0, row_stride) * temp1 + 
                   mgard_common::get_h_cuda(dcoords_y, 0, row_stride) * vec[row_stride * lddv];
    for (int i = row_stride; i < nrow - row_stride; i += row_stride) {
      temp2 = vec[i * lddv];
      vec[i * lddv] = mgard_common::get_h_cuda(dcoords_y, i - row_stride, row_stride) * temp1 + 
               2 * 
                  (mgard_common::get_h_cuda(dcoords_y, i - row_stride, row_stride) +
                   mgard_common::get_h_cuda(dcoords_y, i,              row_stride)) *
               temp2 + 
               mgard_common::get_h_cuda(dcoords_y, i, row_stride) * vec[(i + row_stride)  * lddv];
      temp1 = temp2;
    }
    vec[(nrow-1) * lddv] = mgard_common::get_h_cuda(dcoords_y, nrow - row_stride - 1, row_stride) * temp1 +
              2 * mgard_common::get_h_cuda(dcoords_y, nrow - row_stride - 1, row_stride) * vec[(nrow-1) * lddv];
  }
}

template <typename T>
mgard_cuda_ret 
mass_matrix_multiply_col_cuda(int nrow,       int ncol, 
                              int row_stride, int col_stride,
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

  // std::cout << "threadsPerBlock: " << tb << "\n"; 
  // std::cout << "blockPerGrid: " << grid << "\n"; 
  
  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _mass_matrix_multiply_col_cuda<<<blockPerGrid, threadsPerBlock,
                                   0, stream>>>(nrow,       ncol, 
                                                row_stride, col_stride,
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
mass_matrix_multiply_row_cuda<double>(int nrow,       int ncol, 
                                  int row_stride, int col_stride,
                                  double * dv,    int lddv,
                                  double * dcoords_x,
                                  int B, mgard_cuda_handle & handle, 
                                  int queue_idx, bool profile);
template mgard_cuda_ret 
mass_matrix_multiply_row_cuda<float>(int nrow,       int ncol, 
                                  int row_stride, int col_stride,
                                  float * dv,    int lddv,
                                  float * dcoords_x,
                                  int B, mgard_cuda_handle & handle, 
                                  int queue_idx, bool profile);

template mgard_cuda_ret 
mass_matrix_multiply_col_cuda<double>(int nrow,       int ncol, 
                                  int row_stride, int col_stride,
                                  double * dv,    int lddv,
                                  double * dcoords_y,
                                  int B, mgard_cuda_handle & handle, 
                                  int queue_idx, bool profile);
template mgard_cuda_ret 
mass_matrix_multiply_col_cuda<float>(int nrow,       int ncol, 
                                  int row_stride, int col_stride,
                                  float * dv,    int lddv,
                                  float * dcoords_y,
                                  int B, mgard_cuda_handle & handle, 
                                  int queue_idx, bool profile);

}
}