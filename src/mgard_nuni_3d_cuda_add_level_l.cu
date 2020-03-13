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
_add_level_l_cuda_cpt(int nr,         int nc,         int nf,
                       int row_stride, int col_stride, int fib_stride,
                       T * dv,    int lddv1,      int lddv2,
                       T * dwork, int lddwork1,   int lddwork2) {
  
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int z = z0; z * row_stride < nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y * col_stride < nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x * fib_stride < nf; x += blockDim.x * gridDim.x) {
        int z_strided = z * row_stride;
        int y_strided = y * col_stride;
        int x_strided = x * fib_stride;
        dv[get_idx(lddv1, lddv2, z_strided, y_strided, x_strided)] += dwork[get_idx(lddwork1, lddwork2, z_strided, y_strided, x_strided)];

      }
    }
  }
}

template <typename T>
mgard_cuda_ret 
add_level_l_cuda_cpt(int nr,         int nc,         int nf,
                     int row_stride, int col_stride, int fib_stride,
                     T * dv,    int lddv1,      int lddv2,
                     T * dwork, int lddwork1,   int lddwork2,
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

  _add_level_l_cuda_cpt<<<blockPerGrid, threadsPerBlock,
                          0, stream>>>(nr,         nc,         nf,
                                       row_stride, col_stride, fib_stride,
                                       dv,         lddv1,      lddv2,
                                       dwork,      lddwork1,   lddwork2);

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
add_level_l_cuda_cpt<double>(int nr,         int nc,         int nf,
                             int row_stride, int col_stride, int fib_stride,
                             double * dv,    int lddv1,      int lddv2,
                             double * dwork, int lddwork1,   int lddwork2,
                             int B, mgard_cuda_handle & handle, 
                             int queue_idx, bool profile);
template mgard_cuda_ret 
add_level_l_cuda_cpt<float>(int nr,         int nc,         int nf,
                             int row_stride, int col_stride, int fib_stride,
                             float * dv,    int lddv1,      int lddv2,
                             float * dwork, int lddwork1,   int lddwork2,
                             int B, mgard_cuda_handle & handle, 
                             int queue_idx, bool profile);


}
