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
                            T num) {

  int B = 4;
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

  // std::cout << "_copy_level_l_cuda" << std::endl;
  // std::cout << "thread block: " << tby << ", " << tbx << std::endl;
  // std::cout << "grid: " << gridy << ", " << gridx << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  _assign_num_level_l_cuda_cpt<<<blockPerGrid, threadsPerBlock>>>(nr,         nc,         nf,
                                                                  row_stride, col_stride, fib_stride,
                                                                  dwork,      lddwork1,   lddwork2,
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

template mgard_cuda_ret 
assign_num_level_l_cuda_cpt<double>(int nr,         int nc,         int nf,
                                    int row_stride, int col_stride, int fib_stride, 
                                    double * dwork, int lddwork1,   int lddwork2,
                                    double num);
template mgard_cuda_ret 
assign_num_level_l_cuda_cpt<float>(int nr,         int nc,         int nf,
                                    int row_stride, int col_stride, int fib_stride, 
                                    float * dwork, int lddwork1,   int lddwork2,
                                    float num);


}