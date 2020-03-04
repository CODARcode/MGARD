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

__global__ void 
_copy_level_l_cuda_l2_sm(int nr,             int nc,
                         int row_stride,     int col_stride,
                         double * dv,        int lddv,
                         double * dwork,     int lddwork) {
  
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;

  for (int y = y0; y < nr; y += blockDim.y * gridDim.y * row_stride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x * col_stride) {
      dwork[get_idx(lddwork, y, x)] = dv[get_idx(lddv, y, x)];
    }
  }
}

mgard_cuda_ret 
copy_level_l_cuda_l2_sm(int nr,         int nc,
                        int row_stride, int col_stride,
                        double * dv,    int lddv,
                        double * dwork, int lddwork) {

  int B = 16;
  int total_thread_y = ceil((double)nr/(row_stride));
  int total_thread_x = ceil((double)nc/(col_stride));
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "_copy_level_l_cuda" << std::endl;
  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _copy_level_l_cuda_l2_sm<<<blockPerGrid, threadsPerBlock>>>(nr,         nc,
                                                              row_stride, col_stride,
                                                              dv,         lddv,
                                                              dwork,      lddwork);

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
}