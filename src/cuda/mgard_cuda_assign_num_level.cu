#include "cuda/mgard_cuda_assign_num_level.h"
#include "cuda/mgard_cuda_common_internal.h"

namespace mgard_cuda {

template <typename T>
__global__ void _assign_num_level(int nrow, int ncol, int nr, int nc,
                                  int row_stride, int col_stride, int *dirow,
                                  int *dicol, T num, T *dv, int lddv) {

  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
  for (int y = y0; y < nr; y += blockDim.y * gridDim.y * row_stride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x * col_stride) {
      dv[get_idx(lddv, dirow[y], dicol[x])] = num;
    }
  }
}

template <typename T>
void assign_num_level(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                      int nc, int row_stride, int col_stride, int *dirow,
                      int *dicol, T num, T *dv, int lddv, int queue_idx) {

  int total_thread_y = ceil((float)nr / (row_stride));
  int total_thread_x = ceil((float)nc / (col_stride));
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _assign_num_level<<<blockPerGrid, threadsPerBlock, 0,
                      *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nr, nc, row_stride, col_stride, dirow, dicol, num, dv, lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void assign_num_level<double>(mgard_cuda_handle<double> &handle,
                                       int nrow, int ncol, int nr, int nc,
                                       int row_stride, int col_stride,
                                       int *dirow, int *dicol, double num,
                                       double *dv, int lddv, int queue_idx);
template void assign_num_level<float>(mgard_cuda_handle<float> &handle,
                                      int nrow, int ncol, int nr, int nc,
                                      int row_stride, int col_stride,
                                      int *dirow, int *dicol, float num,
                                      float *dv, int lddv, int queue_idx);

template <typename T>
__global__ void _assign_num_level_cpt(int nr, int nc, int row_stride,
                                      int col_stride, T num, T *dv, int lddv) {

  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * row_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * col_stride;
  for (int y = y0; y < nr; y += blockDim.y * gridDim.y * row_stride) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x * col_stride) {
      dv[get_idx(lddv, y, x)] = num;
    }
  }
}

template <typename T>
void assign_num_level_cpt(mgard_cuda_handle<T> &handle, int nr, int nc,
                          int row_stride, int col_stride, T num, T *dv,
                          int lddv, int queue_idx) {

  int total_thread_y = ceil((float)nr / (row_stride));
  int total_thread_x = ceil((float)nc / (col_stride));
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _assign_num_level_cpt<<<blockPerGrid, threadsPerBlock, 0,
                          *(cudaStream_t *)handle.get(queue_idx)>>>(
      nr, nc, row_stride, col_stride, num, dv, lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void assign_num_level_cpt<double>(mgard_cuda_handle<double> &handle,
                                           int nr, int nc, int row_stride,
                                           int col_stride, double num,
                                           double *dv, int lddv, int queue_idx);
template void assign_num_level_cpt<float>(mgard_cuda_handle<float> &handle,
                                          int nr, int nc, int row_stride,
                                          int col_stride, float num, float *dv,
                                          int lddv, int queue_idx);

template <typename T>
__global__ void _assign_num_level(int nrow, int ncol, int nfib, int nr, int nc,
                                  int nf, int row_stride, int col_stride,
                                  int fib_stride, int *irow, int *icol,
                                  int *ifib, T num, T *dwork, int lddwork1,
                                  int lddwork2) {

  int z0 = (blockIdx.z * blockDim.z + threadIdx.z) * row_stride;
  int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * col_stride;
  int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * fib_stride;
  for (int z = z0; z < nr; z += blockDim.z * gridDim.z * row_stride) {
    for (int y = y0; y < nc; y += blockDim.y * gridDim.y * col_stride) {
      for (int x = x0; x < nf; x += blockDim.x * gridDim.x * fib_stride) {
        dwork[get_idx(lddwork1, lddwork2, irow[z], icol[y], ifib[x])] = num;
      }
    }
  }
}

template <typename T>
void assign_num_level(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                      int nfib, int nr, int nc, int nf, int row_stride,
                      int col_stride, int fib_stride, int *irow, int *icol,
                      int *ifib, T num, T *dwork, int lddwork1, int lddwork2,
                      int queue_idx) {
  int B_adjusted = min(8, handle.B);
  int total_thread_z = ceil((double)nr / (row_stride));
  int total_thread_y = ceil((double)nc / (col_stride));
  int total_thread_x = ceil((double)nf / (fib_stride));
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z / tbz);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _assign_num_level<<<blockPerGrid, threadsPerBlock, 0,
                      *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nfib, nr, nc, nf, row_stride, col_stride, fib_stride, irow,
      icol, ifib, num, dwork, lddwork1, lddwork2);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void assign_num_level<double>(mgard_cuda_handle<double> &handle,
                                       int nrow, int ncol, int nfib, int nr,
                                       int nc, int nf, int row_stride,
                                       int col_stride, int fib_stride,
                                       int *irow, int *icol, int *ifib,
                                       double num, double *dwork, int lddwork1,
                                       int lddwork2, int queue_idx);
template void assign_num_level<float>(mgard_cuda_handle<float> &handle,
                                      int nrow, int ncol, int nfib, int nr,
                                      int nc, int nf, int row_stride,
                                      int col_stride, int fib_stride, int *irow,
                                      int *icol, int *ifib, float num,
                                      float *dwork, int lddwork1, int lddwork2,
                                      int queue_idx);

template <typename T>
__global__ void _assign_num_level_cpt(int nr, int nc, int nf, int row_stride,
                                      int col_stride, int fib_stride, T num,
                                      T *dwork, int lddwork1, int lddwork2) {

  int z0 = blockIdx.z * blockDim.z + threadIdx.z;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int z = z0; z * row_stride < nr; z += blockDim.z * gridDim.z) {
    for (int y = y0; y * col_stride < nc; y += blockDim.y * gridDim.y) {
      for (int x = x0; x * fib_stride < nf; x += blockDim.x * gridDim.x) {
        int z_strided = z * row_stride;
        int y_strided = y * col_stride;
        int x_strided = x * fib_stride;
        dwork[get_idx(lddwork1, lddwork2, z_strided, y_strided, x_strided)] =
            num;
      }
    }
  }
}

template <typename T>
void assign_num_level_cpt(mgard_cuda_handle<T> &handle, int nr, int nc, int nf,
                          int row_stride, int col_stride, int fib_stride, T num,
                          T *dwork, int lddwork1, int lddwork2, int queue_idx) {
  int B_adjusted = min(8, handle.B);
  int total_thread_z = ceil((double)nr / (row_stride));
  int total_thread_y = ceil((double)nc / (col_stride));
  int total_thread_x = ceil((double)nf / (fib_stride));
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z / tbz);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _assign_num_level_cpt<<<blockPerGrid, threadsPerBlock, 0,
                          *(cudaStream_t *)handle.get(queue_idx)>>>(
      nr, nc, nf, row_stride, col_stride, fib_stride, num, dwork, lddwork1,
      lddwork2);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void assign_num_level_cpt<double>(mgard_cuda_handle<double> &handle,
                                           int nr, int nc, int nf,
                                           int row_stride, int col_stride,
                                           int fib_stride, double num,
                                           double *dwork, int lddwork1,
                                           int lddwork2, int queue_idx);
template void assign_num_level_cpt<float>(mgard_cuda_handle<float> &handle,
                                          int nr, int nc, int nf,
                                          int row_stride, int col_stride,
                                          int fib_stride, float num,
                                          float *dwork, int lddwork1,
                                          int lddwork2, int queue_idx);
} // namespace mgard_cuda