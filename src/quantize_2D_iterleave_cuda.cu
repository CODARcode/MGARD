#include "mgard_cuda_helper.h"
#include "mgard_cuda_helper_internal.h"

namespace mgard
{

template <typename T>
__global__ void 
_quantize_2D_iterleave_cuda(int nrow,    int ncol, 
                            T * dv, int lddv, 
                            int * dwork, int lddwork,
                            T quantizer) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int size_ratio = sizeof(T) / sizeof(int);
  for (int y = y0; y < nrow; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < ncol; x += blockDim.x * gridDim.x) {
      register T v = dv[get_idx(lddv, y, x)];
      register T v2 = v/quantizer;
      int quantum = (int)(v2);
      //quantum /= quantizer;
      dwork[get_idx(lddwork, y, x) + size_ratio] = quantum;
      //dwork[get_idx(lddwork, y, x)] = quantum;
    }
  }
}

template <typename T>
mgard_cuda_ret 
quantize_2D_iterleave_cuda (int nrow,    int ncol, 
                           T * dv,  int lddv, 
                           int * dwork,  int lddwork, 
                           T norm, T tol,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile) {

  int size_ratio = sizeof(T) / sizeof(int);
  T quantizer = norm * tol;

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  int total_thread_x = ncol;
  int total_thread_y = nrow;
  int tbx = min(B, total_thread_x);
  int tby = min(B, total_thread_y);
  int gridx = ceil((float)total_thread_x/tbx);
  int gridy = ceil((float)total_thread_y/tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  if (profile) {
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, stream));
  }

  cudaMemcpyAsyncHelper(dwork, &quantizer, sizeof(double), H2D,
                        handle, queue_idx, profile);
  _quantize_2D_iterleave_cuda<<<blockPerGrid, threadsPerBlock,
                                0, stream>>>(nrow, ncol, 
                                             dv, lddv, 
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
quantize_2D_iterleave_cuda<double>(int nrow,    int ncol, 
                           double * dv,  int lddv, 
                           int * dwork,  int lddwork, 
                           double norm, double tol,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile);
template mgard_cuda_ret 
quantize_2D_iterleave_cuda<float>(int nrow,    int ncol, 
                           float * dv,  int lddv, 
                           int * dwork,  int lddwork, 
                           float norm, float tol,
                           int B, mgard_cuda_handle & handle, 
                           int queue_idx, bool profile);

}