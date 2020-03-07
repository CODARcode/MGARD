#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_compact_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>
#include <cmath>

namespace mgard_gen {  

__global__ void 
_copy_level_l_cuda_cpt(int nf,         int nr,         int nc,
                       int fib_stride, int row_stride, int col_stride,
                       double * dv,    int lddv1,      int lddv2,
                       double * dwork, int lddwork1,   int lddwork2) {
  
  int z0 = (blockIdx.z * blockDim.z + threadIdx.z) * fib_stride;
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
  for (int z = z0; z * fib_stride < nf; z += blockDim.z * gridDim.z) {
    for (int y = y0; y * row_stride < nr; y += blockDim.y * gridDim.y) {
      for (int x = x0; x * col_stride < nc; x += blockDim.x * gridDim.x) {
        int z_strided = z * fib_stride;
        int y_strided = y * row_stride;
        int x_strided = x * col_stride;
        dwork[get_idx(lddcv1, lddcv2, z, y, x)] = dv[get_idx(lddv1, lddv2, z_strided, y_strided, x_strided)];

      }
    }
  }
}

mgard_cuda_ret 
copy_level_l_cuda_cpt(int nf,         int nr,         int nc,
                      int fib_stride, int row_stride, int col_stride,
                      double * dv,    int lddv1,      int lddv2,
                      double * dwork, int lddwork1,   int lddwork2) {

  int B = 16;
  int total_thread_z = ceil((double)nf/(fib_stride));
  int total_thread_y = ceil((double)nr/(row_stride));
  int total_thread_x = ceil((double)nc/(col_stride));
  int tbz = min(B, total_thread_z);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  // std::cout << "_copy_level_l_cuda" << std::endl;
  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _copy_level_l_cuda_cpt<<<blockPerGrid, threadsPerBlock>>>(nf,         nr,         nc,
                                                            fib_stride, row_stride, col_stride,
                                                            difib,      dirow,      dicol,
                                                            dv,         lddv1,      lddv2,
                                                            dwork,      lddwork1,   lddwork2);

  gpuErrchk(cudaGetLastError ());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return mgard_cuda_ret(0, milliseconds/1000.0);
}


}
