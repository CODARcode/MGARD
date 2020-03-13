#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"

namespace mgard
{
template <typename T>
__global__ void 
_dequantize_2D_iterleave_cuda(int nrow,    int ncol, 
                             T * dv, int lddv, 
                             int * dwork, int lddwork,
                             T quantizer) {
    int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = blockIdx.y * blockDim.y + threadIdx.y;

    int size_ratio = sizeof(T) / sizeof(int);

    for (int x = x0; x < nrow; x += blockDim.x * gridDim.x) {
        for (int y = y0; y < ncol; y += blockDim.y * gridDim.y) {
            dv[get_idx(lddv, x, y)] = quantizer * (T)(dwork[get_idx(lddwork, x, y) + size_ratio]);
        }
    }
}

template <typename T>
mgard_cuda_ret  
dequantize_2D_iterleave_cuda (int nrow,   int ncol, 
                             T * dv, int lddv, 
                             int * dwork, int lddwork,
                             int B, mgard_cuda_handle & handle, 
                             int queue_idx, bool profile) {

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int size_ratio = sizeof(T) / sizeof(int);
  T quantizer;

  int total_thread_x = nrow;
  int total_thread_y = ncol;
  int tbx = min(B, total_thread_x);
  int tby = min(B, total_thread_y);
  int gridx = ceil(total_thread_x/tbx);
  int gridy = ceil(total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }


  cudaMemcpyAsyncHelper(&quantizer, dwork, sizeof(double), D2H,
                      handle, queue_idx, profile);
  _dequantize_2D_iterleave_cuda<<<blockPerGrid, threadsPerBlock,
                                  0, stream>>>(nrow,  ncol, 
                                              dv,    lddv, 
                                              dwork, lddwork, 
                                              quantizer);

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
dequantize_2D_iterleave_cuda<double>(int nrow,   int ncol, 
                             double * dv, int lddv, 
                             int * dwork, int lddwork,
                             int B, mgard_cuda_handle & handle, 
                             int queue_idx, bool profile);
template mgard_cuda_ret  
dequantize_2D_iterleave_cuda<float>(int nrow,   int ncol, 
                             float * dv, int lddv, 
                             int * dwork, int lddwork,
                             int B, mgard_cuda_handle & handle, 
                             int queue_idx, bool profile);


}