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
_add_level_l_cuda_l2_sm(int nr,          int nc,
                        int row_stride, int col_stride,
                        double * dv,    int lddv, 
                        double * dwork, int lddwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int idx_x = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
    int idx_y = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int y = idx_y; y < nr; y += blockDim.y * gridDim.y * row_stride) {
      for (int x = idx_x; x < nc; x += blockDim.x * gridDim.x * col_stride) {
        dv[get_idx(lddv, y, x)] += dwork[get_idx(lddwork, y, x)];
        //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
        //y += blockDim.y * gridDim.y * stride;
      }
        //x += blockDim.x * gridDim.x * stride;
    }
}

mgard_cuda_ret
add_level_l_cuda_l2_sm(int nr,         int nc,
                       int row_stride, int col_stride,
                       double * dv,    int lddv, 
                       double * dwork, int lddwork) {

  int B = 16;
  int total_thread_x = nc/col_stride;
  int total_thread_y = nr/row_stride;
  int tbx = min(B, total_thread_x);
  int tby = min(B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  // std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
  // std::cout << "grid: " << gridx << ", " << gridy <<std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  _add_level_l_cuda_l2_sm<<<blockPerGrid, threadsPerBlock>>>(nr,         nc,
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