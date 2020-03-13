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
_copy_level_cuda(int nrow,       int ncol,
                int row_stride, int col_stride, 
                T * dv,    int lddv, 
                T * dwork, int lddwork) {
    //int stride = pow (2, l); // current stride
    //int Cstride = stride * 2; // coarser stride
    int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * row_stride;
    int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * col_stride;
    //printf("x = %d, y = %d, stride = %d \n", x,y,stride);
    for (int x = x0; x < nrow; x += blockDim.x * gridDim.x * row_stride) {
        for (int y = y0; y < ncol; y += blockDim.y * gridDim.y * col_stride) {
            
            dwork[get_idx(lddv, x, y)] = dv[get_idx(lddwork, x, y)];
            //printf("x = %d, y = %d, stride = %d, v = %f \n", x,y,stride, work[get_idx(ncol, x, y)]);
            //y += blockDim.y * gridDim.y * stride;
        }
        //x += blockDim.x * gridDim.x * stride;
    }
}


template <typename T>
mgard_cuda_ret 
copy_level_cuda(int nrow,       int ncol, 
                int row_stride, int col_stride,
                T * dv,    int lddv,
                T * dwork, int lddwork,
                int B, mgard_cuda_handle & handle, 
                int queue_idx, bool profile) {
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_y = ceil((float)nrow/row_stride);
  int total_thread_x = ceil((float)ncol/col_stride);
  int tby = min(B, total_thread_y);
  int tbx = min(B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tbx);
  int gridx = ceil((float)total_thread_x/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  _copy_level_cuda<<<blockPerGrid, threadsPerBlock,
                     0, stream>>>(nrow,       ncol, 
                                  row_stride, col_stride, 
                                  dv,         lddv, 
                                  dwork,      lddwork);
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
copy_level_cuda<double>(int nrow,       int ncol, 
                        int row_stride, int col_stride,
                        double * dv,    int lddv,
                        double * dwork, int lddwork,
                        int B, mgard_cuda_handle & handle, 
                        int queue_idx, bool profile);
template mgard_cuda_ret 
copy_level_cuda<float>(int nrow,       int ncol, 
                        int row_stride, int col_stride,
                        float * dv,    int lddv,
                        float * dwork, int lddwork, 
                        int B, mgard_cuda_handle & handle, 
                        int queue_idx, bool profile);



}
}