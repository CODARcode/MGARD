#include "cuda/mgard_cuda_copy_level.h"
#include "cuda/mgard_cuda_common_internal.h"

namespace mgard_cuda {

template <typename T>
__global__ void 
_copy_level(int nrow,           int ncol,
            int nr,             int nc,
            int row_stride,     int col_stride,
            int * irow,         int * icol,
            T * dv,             int lddv,
            T * dwork,          int lddwork) {
  
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
  for (int y = y0; y < nr; y += blockDim.y * gridDim.y * row_stride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x * col_stride) {
      dwork[get_idx(lddwork, irow[y], icol[x])] = dv[get_idx(lddv, irow[y], icol[x])];
    }
  }
}

template <typename T>
void 
copy_level(mgard_cuda_handle<T> & handle, 
                int nrow,       int ncol,
                int nr,         int nc,
                int row_stride, int col_stride,
                int * dirow,    int * dicol,
                T * dv,    int lddv,
                T * dwork, int lddwork,
                int queue_idx) {

  int total_thread_y = ceil((double)nr/(row_stride));
  int total_thread_x = ceil((double)nc/(col_stride));
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _copy_level<<<blockPerGrid, threadsPerBlock,
                0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                    nrow,       ncol,
                                    nr,         nc,
                                    row_stride, col_stride,
                                    dirow,      dicol,
                                    dv,         lddv,
                                    dwork,      lddwork);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
copy_level<double>(mgard_cuda_handle<double> & handle, 
                   int nrow,       int ncol,
                   int nr,         int nc,
                   int row_stride, int col_stride,
                   int * dirow,    int * dicol,
                   double * dv,    int lddv,
                   double * dwork, int lddwork,
                   int queue_idx);
template void 
copy_level<float>(mgard_cuda_handle<float> & handle, 
                  int nrow,       int ncol,
                  int nr,         int nc,
                  int row_stride, int col_stride,
                  int * dirow,    int * dicol,
                  float * dv,     int lddv,
                  float * dwork,  int lddwork,
                  int queue_idx);

template <typename T>
__global__ void 
_copy_level_cpt(int nr,             int nc,
                int row_stride,     int col_stride,
                T * dv,        int lddv,
                T * dwork,     int lddwork) {
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
  for (int y = y0; y < nr; y += blockDim.y * gridDim.y * row_stride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x * col_stride) {
      dwork[get_idx(lddwork, y, x)] = dv[get_idx(lddv, y, x)];
    }
  }
}

template <typename T>
void 
copy_level_cpt(mgard_cuda_handle<T> & handle, 
               int nr,         int nc,
               int row_stride, int col_stride,
               T * dv,    int lddv,
               T * dwork, int lddwork,
               int queue_idx) {

  int total_thread_y = ceil((double)nr/(row_stride));
  int total_thread_x = ceil((double)nc/(col_stride));
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _copy_level_cpt<<<blockPerGrid, threadsPerBlock,
                    0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                          nr,         nc,
                                          row_stride, col_stride,
                                          dv,         lddv,
                                          dwork,      lddwork);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
copy_level_cpt<double>(mgard_cuda_handle<double> & handle,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       double * dv,    int lddv,
                       double * dwork, int lddwork,
                       int queue_idx);
template void 
copy_level_cpt<float>(mgard_cuda_handle<float> & handle,
                       int nr,         int nc,
                       int row_stride, int col_stride,
                       float * dv,     int lddv,
                       float * dwork,  int lddwork,
                       int queue_idx);

template <typename T>
__global__ void  
_copy_level_cpt(int nr,         int nc,         int nf,
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
        dwork[get_idx(lddwork1, lddwork2, z_strided, y_strided, x_strided)] = dv[get_idx(lddv1, lddv2, z_strided, y_strided, x_strided)];

      }
    }
  }
}

template <typename T>
void 
copy_level_cpt(mgard_cuda_handle<T> & handle, 
               int nr,         int nc,         int nf,
               int row_stride, int col_stride, int fib_stride,
               T * dv,    int lddv1,      int lddv2,
               T * dwork, int lddwork1,   int lddwork2,
               int queue_idx) {

  int B_adjusted = min(8, handle.B);   
  int total_thread_z = ceil((double)nr/(row_stride));
  int total_thread_y = ceil((double)nc/(col_stride));
  int total_thread_x = ceil((double)nf/(fib_stride));
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z/tbz);
  int gridy = ceil((float)total_thread_y/tby);
  int gridx = ceil((float)total_thread_x/tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _copy_level_cpt<<<blockPerGrid, threadsPerBlock,
                    0, *(cudaStream_t *)handle.get(queue_idx)>>>(
                                        nr,         nc,         nf,
                                        row_stride, col_stride, fib_stride,
                                        dv,         lddv1,      lddv2,
                                        dwork,      lddwork1,   lddwork2);

  gpuErrchk(cudaGetLastError ());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize()); 
#endif
}

template void 
copy_level_cpt<double>(mgard_cuda_handle<double> & handle, 
                       int nr,         int nc,         int nf,
                       int row_stride, int col_stride, int fib_stride,
                       double * dv,    int lddv1,      int lddv2,
                       double * dwork, int lddwork1,   int lddwork2,
                       int queue_idx);
template void 
copy_level_cpt<float>(mgard_cuda_handle<float> & handle, 
                       int nr,         int nc,         int nf,
                       int row_stride, int col_stride, int fib_stride,
                       float * dv,    int lddv1,      int lddv2,
                       float * dwork, int lddwork1,   int lddwork2,
                       int queue_idx);


}
