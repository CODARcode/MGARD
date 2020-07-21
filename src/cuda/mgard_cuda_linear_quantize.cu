#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_linear_quantize.h"
#include <iostream>
namespace mgard_cuda {

template <typename T>
__global__ void _linear_quantize(int nrow, int ncol, T quantizer, T *dv,
                                 int lddv, int *dwork, int lddwork) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int size_ratio = sizeof(T) / sizeof(int);
  for (int y = y0; y < nrow; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < ncol; x += blockDim.x * gridDim.x) {
      // dwork[get_idx(lddwork, y, x) + size_ratio] = (int)(dv[get_idx(lddv, y,
      // x)]/quantizer);
      dwork[get_idx(lddwork, y, x) + size_ratio] =
          copysign(0.5 + fabs(dv[get_idx(lddv, y, x)] / quantizer),
                   dv[get_idx(lddv, y, x)]);
    }
  }
}

template <typename T>
void linear_quantize(mgard_cuda_handle<T> &handle, int nrow, int ncol, T norm,
                     T tol, T *dv, int lddv, int *dwork, int lddwork,
                     int queue_idx) {

  T quantizer = norm * tol;
  int total_thread_x = ncol;
  int total_thread_y = nrow;
  int tbx = min(handle.B, total_thread_x);
  int tby = min(handle.B, total_thread_y);
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  cudaMemcpyAsyncHelper(handle, dwork, &quantizer, sizeof(double), H2D,
                        queue_idx);
  _linear_quantize<<<blockPerGrid, threadsPerBlock, 0,
                     *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, quantizer, dv, lddv, dwork, lddwork);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void linear_quantize<double>(mgard_cuda_handle<double> &handle,
                                      int nrow, int ncol, double norm,
                                      double tol, double *dv, int lddv,
                                      int *dwork, int lddwork, int queue_idx);
template void linear_quantize<float>(mgard_cuda_handle<float> &handle, int nrow,
                                     int ncol, float norm, float tol, float *dv,
                                     int lddv, int *dwork, int lddwork,
                                     int queue_idx);
;

template <typename T>
__global__ void _linear_dequantize(int nrow, int ncol, T quantizer, T *dv,
                                   int lddv, int *dwork, int lddwork) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int size_ratio = sizeof(T) / sizeof(int);
  for (int y = y0; y < nrow; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < ncol; x += blockDim.x * gridDim.x) {
      dv[get_idx(lddv, y, x)] =
          quantizer * (T)(dwork[get_idx(lddwork, y, x) + size_ratio]);
    }
  }
}

template <typename T>
void linear_dequantize(mgard_cuda_handle<T> &handle, int nrow, int ncol, T *dv,
                       int lddv, int *dwork, int lddwork, int queue_idx) {

  T quantizer;
  int total_thread_x = nrow;
  int total_thread_y = ncol;
  int tbx = min(handle.B, total_thread_x);
  int tby = min(handle.B, total_thread_y);
  int gridx = ceil(total_thread_x / tbx);
  int gridy = ceil(total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);

  cudaMemcpyAsyncHelper(handle, &quantizer, dwork, sizeof(double), D2H,
                        queue_idx);
  _linear_dequantize<<<blockPerGrid, threadsPerBlock, 0,
                       *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, quantizer, dv, lddv, dwork, lddwork);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void linear_dequantize<double>(mgard_cuda_handle<double> &handle,
                                        int nrow, int ncol, double *dv,
                                        int lddv, int *dwork, int lddwork,
                                        int queue_idx);
template void linear_dequantize<float>(mgard_cuda_handle<float> &handle,
                                       int nrow, int ncol, float *dv, int lddv,
                                       int *dwork, int lddwork, int queue_idx);

template <typename T>
__global__ void _linear_quantize(int nrow, int ncol, int nfib, T quantizer,
                                 T *v, int ldv1, int ldv2, int *work,
                                 int ldwork1, int ldwork2) {
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int size_ratio = sizeof(T) / sizeof(int);
  for (int z = z0; z < nrow; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < ncol; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nfib; x += blockDim.x * gridDim.x) {
        // dwork[get_idx(lddwork, y, x) + size_ratio] = (int)(dv[get_idx(lddv,
        // y, x)]/quantizer);
        T t = v[get_idx(ldv1, ldv2, z, y, x)];
        work[get_idx(ldwork1, ldwork2, z, y, x) + size_ratio] =
            copysign(0.5 + fabs(t / quantizer), t);
      }
    }
  }
}

template <typename T>
void linear_quantize(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                     T norm, T tol, T *dv, int lddv1, int lddv2, int *dwork,
                     int lddwork1, int lddwork2, int queue_idx) {

  T quantizer = norm * tol;

  int B_adjusted = min(8, handle.B);
  int total_thread_x = nfib;
  int total_thread_y = ncol;
  int total_thread_z = nrow;
  int tbx = min(B_adjusted, total_thread_x);
  int tby = min(B_adjusted, total_thread_y);
  int tbz = min(B_adjusted, total_thread_z);
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  int gridz = ceil((float)total_thread_z / tbz);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  cudaMemcpyAsyncHelper(handle, dwork, &quantizer, sizeof(double), H2D,
                        queue_idx);
  _linear_quantize<<<blockPerGrid, threadsPerBlock, 0,
                     *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nfib, quantizer, dv, lddv1, lddv2, dwork, lddwork1, lddwork2);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void linear_quantize<double>(mgard_cuda_handle<double> &handle,
                                      int nrow, int ncol, int nfib, double norm,
                                      double tol, double *dv, int lddv1,
                                      int lddv2, int *dwork, int lddwork1,
                                      int lddwork2, int queue_idx);
template void linear_quantize<float>(mgard_cuda_handle<float> &handle, int nrow,
                                     int ncol, int nfib, float norm, float tol,
                                     float *dv, int lddv1, int lddv2,
                                     int *dwork, int lddwork1, int lddwork2,
                                     int queue_idx);

template <typename T>
__global__ void _linear_dequantize(int nrow, int ncol, int nfib, T quantizer,
                                   T *v, int ldv1, int ldv2, int *work,
                                   int ldwork1, int ldwork2) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int z0 = blockIdx.z * blockDim.z + threadIdx.z;
  int size_ratio = sizeof(T) / sizeof(int);
  for (int z = z0; z < nrow; z += blockDim.z * gridDim.z) {
    for (int y = y0; y < ncol; y += blockDim.y * gridDim.y) {
      for (int x = x0; x < nfib; x += blockDim.x * gridDim.x) {
        v[get_idx(ldv1, ldv2, z, y, x)] =
            quantizer *
            (T)(work[get_idx(ldwork1, ldwork2, z, y, x) + size_ratio]);
      }
    }
  }
}

template <typename T>
void linear_dequantize(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                       int nfib, T *dv, int lddv1, int lddv2, int *dwork,
                       int lddwork1, int lddwork2, int queue_idx) {

  T quantizer;
  int B_adjusted = min(8, handle.B);
  int total_thread_x = nfib;
  int total_thread_y = ncol;
  int total_thread_z = nrow;
  int tbx = min(B_adjusted, total_thread_x);
  int tby = min(B_adjusted, total_thread_y);
  int tbz = min(B_adjusted, total_thread_z);
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  int gridz = ceil((float)total_thread_z / tbz);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);

  cudaMemcpyAsyncHelper(handle, &quantizer, dwork, sizeof(double), D2H,
                        queue_idx);
  _linear_dequantize<<<blockPerGrid, threadsPerBlock, 0,
                       *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nfib, quantizer, dv, lddv1, lddv2, dwork, lddwork1, lddwork2);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void linear_dequantize<double>(mgard_cuda_handle<double> &handle,
                                        int nrow, int ncol, int nfib,
                                        double *dv, int lddv1, int lddv2,
                                        int *dwork, int lddwork1, int lddwork2,
                                        int queue_idx);
template void linear_dequantize<float>(mgard_cuda_handle<float> &handle,
                                       int nrow, int ncol, int nfib, float *dv,
                                       int lddv1, int lddv2, int *dwork,
                                       int lddwork1, int lddwork2,
                                       int queue_idx);
} // namespace mgard_cuda