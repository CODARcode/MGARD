#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>
#include <cmath>

namespace mgard_gen {  


template <typename T>
__global__ void 
_assign_num_level_l_cuda(int nrow,       int ncol,       int nfib,
                         int nr,         int nc,         int nf,
                         int row_stride, int col_stride, int fib_stride,
                         int * irow,     int * icol,     int * ifib, 
                         T * dwork, int lddwork1,   int lddwork2,
                         T num) {
  
  int z0 = (blockIdx.z * blockDim.z + threadIdx.z) * row_stride;
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * col_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * fib_stride;
  for (int z = z0; z < nr; z += blockDim.z * gridDim.z * row_stride) {
    for (int y = y0; y < nc; y += blockDim.y * gridDim.y * col_stride) {
      for (int x = x0; x < nf; x += blockDim.x * gridDim.x * fib_stride) {
        dwork[get_idx(lddwork1, lddwork2, irow[z], icol[y], ifib[x])] = num;

      }
    }
  }
}

template <typename T>
mgard_cuda_ret 
assign_num_level_l_cuda(int nrow,       int ncol,       int nfib,
                         int nr,         int nc,         int nf,
                         int row_stride, int col_stride, int fib_stride,
                         int * irow,     int * icol,     int * ifib, 
                         T * dwork, int lddwork1,   int lddwork2,
                         T num,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile) {
  B = min(8, B);  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_z = ceil((double)nr/(row_stride));
  int total_thread_y = ceil((double)nc/(col_stride));
  int total_thread_x = ceil((double)nf/(fib_stride));
  int tbz = min(B, total_thread_z);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _assign_num_level_l_cuda<<<blockPerGrid, threadsPerBlock,
                                 0, stream>>>(nrow,       ncol,       nfib,
                                              nr,         nc,         nf,
                                              row_stride, col_stride, fib_stride,
                                              irow,       icol,       ifib,
                                              dwork,      lddwork1,   lddwork2,
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
_assign_num_level_l_cuda_cpt(int nr,         int nc,         int nf,
                             int row_stride, int col_stride, int fib_stride, 
                             T * dwork, int lddwork1,   int lddwork2,
                             T num) {
  
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int z = z0; z * row_stride < nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y * col_stride < nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x * fib_stride < nf; x += blockDim.x * gridDim.x) {
        int z_strided = z * row_stride;
        int y_strided = y * col_stride;
        int x_strided = x * fib_stride;
        dwork[get_idx(lddwork1, lddwork2, z_strided, y_strided, x_strided)] = num;

      }
    }
  }
}

template <typename T>
mgard_cuda_ret 
assign_num_level_l_cuda_cpt(int nr,         int nc,         int nf,
                            int row_stride, int col_stride, int fib_stride, 
                            T * dwork, int lddwork1,   int lddwork2,
                            T num,
                            int B, mgard_cuda_handle & handle, 
                            int queue_idx, bool profile) {
  B = min(8, B);  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_z = ceil((double)nr/(row_stride));
  int total_thread_y = ceil((double)nc/(col_stride));
  int total_thread_x = ceil((double)nf/(fib_stride));
  int tbz = min(B, total_thread_z);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _assign_num_level_l_cuda_cpt<<<blockPerGrid, threadsPerBlock,
                                 0, stream>>>(nr,         nc,         nf,
                                              row_stride, col_stride, fib_stride,
                                              dwork,      lddwork1,   lddwork2,
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
assign_num_level_l_cuda<double>(int nrow,       int ncol,       int nfib,
                         int nr,         int nc,         int nf,
                         int row_stride, int col_stride, int fib_stride,
                         int * irow,     int * icol,     int * ifib, 
                         double * dwork, int lddwork1,   int lddwork2,
                         double num,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);
template mgard_cuda_ret 
assign_num_level_l_cuda<float>(int nrow,       int ncol,       int nfib,
                         int nr,         int nc,         int nf,
                         int row_stride, int col_stride, int fib_stride,
                         int * irow,     int * icol,     int * ifib, 
                         float * dwork, int lddwork1,   int lddwork2,
                         float num,
                         int B, mgard_cuda_handle & handle, 
                         int queue_idx, bool profile);

template mgard_cuda_ret 
assign_num_level_l_cuda_cpt<double>(int nr,         int nc,         int nf,
                                    int row_stride, int col_stride, int fib_stride, 
                                    double * dwork, int lddwork1,   int lddwork2,
                                    double num,
                                    int B, mgard_cuda_handle & handle, 
                                    int queue_idx, bool profile);
template mgard_cuda_ret 
assign_num_level_l_cuda_cpt<float>(int nr,         int nc,         int nf,
                                    int row_stride, int col_stride, int fib_stride, 
                                    float * dwork, int lddwork1,   int lddwork2,
                                    float num,
                                    int B, mgard_cuda_handle & handle, 
                                    int queue_idx, bool profile);


}