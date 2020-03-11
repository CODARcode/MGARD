#include "mgard_nuni.h"
#include "mgard.h"
#include "mgard_nuni_2d_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"
#include <fstream>

namespace mgard_2d {
namespace mgard_cannon {

template <typename T>
__global__ void 
_subtract_level_cuda(int nrow,       int ncol, 
                      int row_stride, int col_stride,
                      T * dv,    int lddv, 
                      T * dwork, int lddwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int idx_x = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
    int idx_y = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int y = idx_y; y < nrow; y += blockDim.y * gridDim.y * row_stride) {
      for (int x = idx_x; x < ncol; x += blockDim.x * gridDim.x * col_stride) {
        dv[get_idx(lddv, y, x)] -= dwork[get_idx(lddwork, y, x)];
        //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
        //y += blockDim.y * gridDim.y * stride;
      }
        //x += blockDim.x * gridDim.x * stride;
    }
}

template <typename T>
mgard_cuda_ret 
subtract_level_cuda(int nrow,       int ncol, 
                    int row_stride, int col_stride,
                    T * dv,    int lddv, 
                    T * dwork, int lddwork) {
  int B = 16;
  int total_thread_x = ncol/col_stride;
  int total_thread_y = nrow/row_stride;
  int tbx = min(B, total_thread_x);
  int tby = min(B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  //std::cout << "thread block: " << tbx << ", " << tby <<std::endl;
  //std::cout << "grid: " << gridx << ", " << gridy <<std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _subtract_level_cuda<<<blockPerGrid, threadsPerBlock>>>(nrow,       ncol,
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

template mgard_cuda_ret 
subtract_level_cuda<double>(int nrow,       int ncol, 
                            int row_stride, int col_stride,
                            double * dv,    int lddv, 
                            double * dwork, int lddwork);
template mgard_cuda_ret 
subtract_level_cuda<float>(int nrow,       int ncol, 
                            int row_stride, int col_stride,
                            float * dv,    int lddv, 
                            float * dwork, int lddwork);

}
}
