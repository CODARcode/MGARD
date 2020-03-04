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
_assign_num_level_l_cuda_l2_sm(int nr,             int nc,
                               int row_stride,     int col_stride,
                               double * dv,        int lddv,
                               double num) {
  
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;

  for (int y = y0; y < nr; y += blockDim.y * gridDim.y * row_stride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x * col_stride) {
      dv[get_idx(lddv, y, x)] = num;
    }
  }
}


mgard_cuda_ret 
assign_num_level_l_cuda_l2_sm(int nr,             int nc,
                              int row_stride,     int col_stride,
                              double * dv,        int lddv,
                              double num) {
  int B = 16;
  int total_thread_y = ceil((float)nr/(row_stride));
  int total_thread_x = ceil((float)nc/(col_stride));
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);


  // std::cout << "thread block: " << tby << ", " << tby << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _assign_num_level_l_cuda_l2_sm<<<blockPerGrid, threadsPerBlock>>>(nr,         nc,
                                                                    row_stride, col_stride,
                                                                    dv,         lddv,
                                                                    num);
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