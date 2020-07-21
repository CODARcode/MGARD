#include "cuda/mgard_cuda_common.h"
#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_compact_helper.h"
#include <iomanip>
#include <iostream>

namespace mgard_cuda {
/* 3D Original to (2^k)+1 */
template <typename T>
__global__ void _org_to_pow2p1(int nrow, int ncol, int nfib, int nr, int nc,
                               int nf, int *dirow, int *dicol, int *difib,
                               T *dv, int lddv1, int lddv2, T *dcv, int lddcv1,
                               int lddcv2) {

  int r0 = blockIdx.z * blockDim.z + threadIdx.z;
  int c0 = blockIdx.y * blockDim.y + threadIdx.y;
  int f0 = blockIdx.x * blockDim.x + threadIdx.x;

  for (int r = r0; r < nr; r += blockDim.z * gridDim.z) {
    for (int c = c0; c < nc; c += blockDim.y * gridDim.y) {
      for (int f = f0; f < nf; f += blockDim.x * gridDim.x) {
        dcv[get_idx(lddcv1, lddcv2, r, c, f)] =
            dv[get_idx(lddv1, lddv2, dirow[r], dicol[c], difib[f])];
      }
    }
  }
}

template <typename T>
void org_to_pow2p1(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                   int nr, int nc, int nf, int *dirow, int *dicol, int *difib,
                   T *dv, int lddv1, int lddv2, T *dcv, int lddcv1, int lddcv2,
                   int queue_idx) {

  int B_adjusted = min(8, handle.B);
  int total_thread_z = nr;
  int total_thread_y = nc;
  int total_thread_x = nf;
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z / tbz);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _org_to_pow2p1<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nfib, nr, nc, nf, dirow, dicol, difib, dv, lddv1, lddv2, dcv,
      lddcv1, lddcv2);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void org_to_pow2p1<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int ncol, int nfib, int nr, int nc, int nf,
                                    int *dirow, int *dicol, int *difib,
                                    double *dv, int lddv1, int lddv2,
                                    double *dcv, int lddcv1, int lddcv2,
                                    int queue_idx);
template void org_to_pow2p1<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int ncol, int nfib, int nr, int nc, int nf,
                                   int *dirow, int *dicol, int *difib,
                                   float *dv, int lddv1, int lddv2, float *dcv,
                                   int lddcv1, int lddcv2, int queue_idx);

/* 2D Original to (2^k)+1 */
template <typename T>
__global__ void _org_to_pow2p1(int nrow, int ncol, int nr, int nc, int *dirow,
                               int *dicol, T *dv, int lddv, T *dcv, int lddcv) {

  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int y = y0; y < nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x) {
      dcv[get_idx(lddcv, y, x)] = dv[get_idx(lddv, dirow[y], dicol[x])];
    }
  }
}

template <typename T>
void org_to_pow2p1(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                   int nc, int *dirow, int *dicol, T *dv, int lddv, T *dcv,
                   int lddcv, int queue_idx) {

  int total_thread_y = nr;
  int total_thread_x = nc;
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _org_to_pow2p1<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nr, nc, dirow, dicol, dv, lddv, dcv, lddcv);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void org_to_pow2p1<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int ncol, int nr, int nc, int *dirow,
                                    int *dicol, double *dv, int lddv,
                                    double *dcv, int lddcv, int queue_idx);
template void org_to_pow2p1<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int ncol, int nr, int nc, int *dirow,
                                   int *dicol, float *dv, int lddv, float *dcv,
                                   int lddcv, int queue_idx);

/* 1D Original to (2^k)+1 */
template <typename T>
__global__ void _org_to_pow2p1(int nrow, int nr, int *dirow, T *dv, T *dcv) {

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int x = x0; x < nr; x += blockDim.x * gridDim.x) {
    dcv[x] = dv[dirow[x]];
  }
}

template <typename T>
void org_to_pow2p1(mgard_cuda_handle<T> &handle, int nrow, int nr, int *dirow,
                   T *dv, T *dcv, int queue_idx) {

  int total_thread_y = 1;
  int total_thread_x = nr;
  int tby = 1;
  int tbx = min(handle.B, total_thread_x);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _org_to_pow2p1<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(nrow, nr, dirow,
                                                             dv, dcv);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void org_to_pow2p1<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int nr, int *dirow, double *dv, double *dcv,
                                    int queue_idx);
template void org_to_pow2p1<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int nr, int *dirow, float *dv, float *dcv,
                                   int queue_idx);

/* 3D (2^k)+1 to original*/
template <typename T>
__global__ void _pow2p1_to_org(int nrow, int ncol, int nfib, int nr, int nc,
                               int nf, int *dirow, int *dicol, int *difib,
                               T *dcv, int lddcv1, int lddcv2, T *dv, int lddv1,
                               int lddv2) {

  int r0 = blockIdx.z * blockDim.z + threadIdx.z;
  int c0 = blockIdx.y * blockDim.y + threadIdx.y;
  int f0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int r = r0; r < nr; r += blockDim.z * gridDim.z) {
    for (int c = c0; c < nc; c += blockDim.y * gridDim.y) {
      for (int f = f0; f < nf; f += blockDim.x * gridDim.x) {
        dv[get_idx(lddv1, lddv2, dirow[r], dicol[c], difib[f])] =
            dcv[get_idx(lddcv1, lddcv2, r, c, f)];
      }
    }
  }
}

template <typename T>
void pow2p1_to_org(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                   int nr, int nc, int nf, int *dirow, int *dicol, int *difib,
                   T *dcv, int lddcv1, int lddcv2, T *dv, int lddv1, int lddv2,
                   int queue_idx) {

  int B_adjusted = min(8, handle.B);
  int total_thread_z = nr;
  int total_thread_y = nc;
  int total_thread_x = nf;
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z / tbz);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _pow2p1_to_org<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nfib, nr, nc, nf, dirow, dicol, difib, dcv, lddcv1, lddcv2,
      dv, lddv1, lddv2);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void pow2p1_to_org<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int ncol, int nfib, int nr, int nc, int nf,
                                    int *dirow, int *dicol, int *difib,
                                    double *dcv, int lddcv1, int lddcv2,
                                    double *dv, int lddv1, int lddv2,
                                    int queue_idx);
template void pow2p1_to_org<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int ncol, int nfib, int nr, int nc, int nf,
                                   int *dirow, int *dicol, int *difib,
                                   float *dcv, int lddcv1, int lddcv2,
                                   float *dv, int lddv1, int lddv2,
                                   int queue_idx);

/* 2D (2^k)+1 to original*/
template <typename T>
__global__ void _pow2p1_to_org(int nrow, int ncol, int nr, int nc, int *dirow,
                               int *dicol, T *dcv, int lddcv, T *dv, int lddv) {

  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int y = y0; y < nr; y += blockDim.y * gridDim.y) {
    for (int x = x0; x < nc; x += blockDim.x * gridDim.x) {
      dv[get_idx(lddv, dirow[y], dicol[x])] = dcv[get_idx(lddcv, y, x)];
    }
  }
}

template <typename T>
void pow2p1_to_org(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nr,
                   int nc, int *dirow, int *dicol, T *dcv, int lddcv, T *dv,
                   int lddv, int queue_idx) {

  int total_thread_y = nr;
  int total_thread_x = nc;
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _pow2p1_to_org<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nr, nc, dirow, dicol, dcv, lddcv, dv, lddv);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void pow2p1_to_org<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int ncol, int nr, int nc, int *dirow,
                                    int *dicol, double *dcv, int lddcv,
                                    double *dv, int lddv, int queue_idx);
template void pow2p1_to_org<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int ncol, int nr, int nc, int *dirow,
                                   int *dicol, float *dcv, int lddcv, float *dv,
                                   int lddv, int queue_idx);

/* 1D (2^k)+1 to original */
template <typename T>
__global__ void _pow2p1_to_org(int nrow, int nr, int *dirow, T *dcv, T *dv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int x = x0; x < nr; x += blockDim.x * gridDim.x) {
    dv[dirow[x]] = dcv[x];
  }
}

template <typename T>
void pow2p1_to_org(mgard_cuda_handle<T> &handle, int nrow, int nr, int *dirow,
                   T *dcv, T *dv, int queue_idx) {

  int total_thread_y = 1;
  int total_thread_x = nr;
  int tby = 1;
  int tbx = min(handle.B, total_thread_x);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _pow2p1_to_org<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(nrow, nr, dirow,
                                                             dcv, dv);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void pow2p1_to_org<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int nr, int *dirow, double *dcv, double *dv,
                                    int queue_idx);
template void pow2p1_to_org<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int nr, int *dirow, float *dcv, float *dv,
                                   int queue_idx);

/* 3D (2^k)+1 to compact */
template <typename T>
__global__ void _pow2p1_to_cpt(int nrow, int ncol, int nfib, int row_stride,
                               int col_stride, int fib_stride, T *dv, int lddv1,
                               int lddv2, T *dcv, int lddcv1, int lddcv2) {
  int r0 = blockIdx.z * blockDim.z + threadIdx.z;
  int c0 = blockIdx.y * blockDim.y + threadIdx.y;
  int f0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int r = r0; r * row_stride < nrow; r += blockDim.z * gridDim.z) {
    for (int c = c0; c * col_stride < ncol; c += blockDim.y * gridDim.y) {
      for (int f = f0; f * fib_stride < nfib; f += blockDim.x * gridDim.x) {
        int r_strided = r * row_stride;
        int c_strided = c * col_stride;
        int f_strided = f * fib_stride;
        dcv[get_idx(lddcv1, lddcv2, r, c, f)] =
            dv[get_idx(lddv1, lddv2, r_strided, c_strided, f_strided)];
      }
    }
  }
}

template <typename T>
void pow2p1_to_cpt(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                   int row_stride, int col_stride, int fib_stride, T *dv,
                   int lddv1, int lddv2, T *dcv, int lddcv1, int lddcv2,
                   int queue_idx) {

  int B_adjusted = min(8, handle.B);
  int total_thread_z = ceil((float)nrow / row_stride);
  int total_thread_y = ceil((float)ncol / col_stride);
  int total_thread_x = ceil((float)nfib / fib_stride);
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z / tbz);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _pow2p1_to_cpt<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nfib, row_stride, col_stride, fib_stride, dv, lddv1, lddv2,
      dcv, lddcv1, lddcv2);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void pow2p1_to_cpt<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int ncol, int nfib, int row_stride,
                                    int col_stride, int fib_stride, double *dv,
                                    int lddv1, int lddv2, double *dcv,
                                    int lddcv1, int lddcv2, int queue_idx);
template void pow2p1_to_cpt<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int ncol, int nfib, int row_stride,
                                   int col_stride, int fib_stride, float *dv,
                                   int lddv1, int lddv2, float *dcv, int lddcv1,
                                   int lddcv2, int queue_idx);

/* 2D (2^k)+1 to compact */
template <typename T>
__global__ void _pow2p1_to_cpt(int nrow, int ncol, int row_stride,
                               int col_stride, T *dv, int lddv, T *dcv,
                               int lddcv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  for (int y = y0; y * row_stride < nrow; y += blockDim.y * gridDim.y) {
    for (int x = x0; x * col_stride < ncol; x += blockDim.x * gridDim.x) {
      int x_strided = x * col_stride;
      int y_strided = y * row_stride;
      // printf("load dv[%d, %d] = %f\n", y_strided, x_strided, dv[get_idx(lddv,
      // y_strided, x_strided)]);
      dcv[get_idx(lddcv, y, x)] = dv[get_idx(lddv, y_strided, x_strided)];
    }
  }
}

template <typename T>
void pow2p1_to_cpt(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                   int row_stride, int col_stride, T *dv, int lddv, T *dcv,
                   int lddcv, int queue_idx) {

  int total_thread_y = ceil((float)nrow / row_stride);
  int total_thread_x = ceil((float)ncol / col_stride);
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _pow2p1_to_cpt<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, row_stride, col_stride, dv, lddv, dcv, lddcv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void pow2p1_to_cpt<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int ncol, int row_stride, int col_stride,
                                    double *dv, int lddv, double *dcv,
                                    int lddcv, int queue_idx);
template void pow2p1_to_cpt<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int ncol, int row_stride, int col_stride,
                                   float *dv, int lddv, float *dcv, int lddcv,
                                   int queue_idx);

/* 2D (2^k)+1 to compact with number assign */
template <typename T>
__global__ void _pow2p1_to_cpt_num_assign(int nrow, int ncol, int row_stride,
                                          int col_stride, T val, T *dv,
                                          int lddv, T *dcv, int lddcv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  for (int y = y0; y * row_stride < nrow; y += blockDim.y * gridDim.y) {
    for (int x = x0; x * col_stride < ncol; x += blockDim.x * gridDim.x) {
      int x_strided = x * col_stride;
      int y_strided = y * row_stride;
      if (y % 2 == 0 && x % 2 == 0)
        dcv[get_idx(lddcv, y, x)] = val;
      else
        dcv[get_idx(lddcv, y, x)] = dv[get_idx(lddv, y_strided, x_strided)];
    }
  }
}

template <typename T>
void pow2p1_to_cpt_num_assign(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                              int row_stride, int col_stride, T val, T *dv,
                              int lddv, T *dcv, int lddcv, int queue_idx) {

  int total_thread_y = ceil((float)nrow / row_stride);
  int total_thread_x = ceil((float)ncol / col_stride);
  int tby = min(handle.B, total_thread_y);
  int tbx = min(handle.B, total_thread_x);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _pow2p1_to_cpt_num_assign<<<blockPerGrid, threadsPerBlock, 0,
                              *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, row_stride, col_stride, val, dv, lddv, dcv, lddcv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void
pow2p1_to_cpt_num_assign<double>(mgard_cuda_handle<double> &handle, int nrow,
                                 int ncol, int row_stride, int col_stride,
                                 double val, double *dv, int lddv, double *dcv,
                                 int lddcv, int queue_idx);
template void pow2p1_to_cpt_num_assign<float>(mgard_cuda_handle<float> &handle,
                                              int nrow, int ncol,
                                              int row_stride, int col_stride,
                                              float val, float *dv, int lddv,
                                              float *dcv, int lddcv,
                                              int queue_idx);

/* 1D (2^k)+1 to compact */
template <typename T>
__global__ void _pow2p1_to_cpt(int nrow, int row_stride, T *dv, T *dcv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int x = x0; x * row_stride < nrow; x += blockDim.x * gridDim.x) {
    int x_strided = x * row_stride;
    dcv[x] = dv[x_strided];
  }
}

template <typename T>
void pow2p1_to_cpt(mgard_cuda_handle<T> &handle, int nrow, int row_stride,
                   T *dv, T *dcv, int queue_idx) {

  int total_thread_y = 1;
  int total_thread_x = ceil((float)nrow / row_stride);
  int tby = 1;
  int tbx = min(handle.B, total_thread_x);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _pow2p1_to_cpt<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(nrow, row_stride,
                                                             dv, dcv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void pow2p1_to_cpt<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int row_stride, double *dv, double *dcv,
                                    int queue_idx);
template void pow2p1_to_cpt<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int row_stride, float *dv, float *dcv,
                                   int queue_idx);

/* 3D compact to (2^k)+1*/
template <typename T>
__global__ void _cpt_to_pow2p1(int nrow, int ncol, int nfib, int row_stride,
                               int col_stride, int fib_stride, T *dcv,
                               int lddcv1, int lddcv2, T *dv, int lddv1,
                               int lddv2) {
  int r0 = blockIdx.z * blockDim.z + threadIdx.z;
  int c0 = blockIdx.y * blockDim.y + threadIdx.y;
  int f0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int r = r0; r * row_stride < nrow; r += blockDim.z * gridDim.z) {
    for (int c = c0; c * col_stride < ncol; c += blockDim.y * gridDim.y) {
      for (int f = f0; f * fib_stride < nfib; f += blockDim.x * gridDim.x) {
        int r_strided = r * row_stride;
        int c_strided = c * col_stride;
        int f_strided = f * fib_stride;
        dv[get_idx(lddv1, lddv2, r_strided, c_strided, f_strided)] =
            dcv[get_idx(lddcv1, lddcv2, r, c, f)];
      }
    }
  }
}

template <typename T>
void cpt_to_pow2p1(mgard_cuda_handle<T> &handle, int nrow, int ncol, int nfib,
                   int row_stride, int col_stride, int fib_stride, T *dcv,
                   int lddcv1, int lddcv2, T *dv, int lddv1, int lddv2,
                   int queue_idx) {

  int B_adjusted = min(8, handle.B);
  int total_thread_z = ceil((float)nrow / row_stride);
  int total_thread_y = ceil((float)ncol / col_stride);
  int total_thread_x = ceil((float)nfib / fib_stride);
  int tbz = min(B_adjusted, total_thread_z);
  int tby = min(B_adjusted, total_thread_y);
  int tbx = min(B_adjusted, total_thread_x);
  int gridz = ceil((float)total_thread_z / tbz);
  int gridy = ceil((float)total_thread_y / tby);
  int gridx = ceil((float)total_thread_x / tbx);
  dim3 threadsPerBlock(tbx, tby, tbz);
  dim3 blockPerGrid(gridx, gridy, gridz);
  _cpt_to_pow2p1<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, nfib, row_stride, col_stride, fib_stride, dcv, lddcv1, lddcv2,
      dv, lddv1, lddv2);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void cpt_to_pow2p1<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int ncol, int nfib, int row_stride,
                                    int col_stride, int fib_stride, double *dcv,
                                    int lddcv1, int lddcv2, double *dv,
                                    int lddv1, int lddv2, int queue_idx);
template void cpt_to_pow2p1<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int ncol, int nfib, int row_stride,
                                   int col_stride, int fib_stride, float *dcv,
                                   int lddcv1, int lddcv2, float *dv, int lddv1,
                                   int lddv2, int queue_idx);

/* 2D compact to (2^k)+1*/
template <typename T>
__global__ void _cpt_to_pow2p1(int nrow, int ncol, int row_stride,
                               int col_stride, T *dcv, int lddcv, T *dv,
                               int lddv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  for (int y = y0; y * row_stride < nrow; y += blockDim.y * gridDim.y) {
    for (int x = x0; x * col_stride < ncol; x += blockDim.x * gridDim.x) {
      int x_strided = x * col_stride;
      int y_strided = y * row_stride;
      dv[get_idx(lddv, y_strided, x_strided)] = dcv[get_idx(lddcv, y, x)];
    }
  }
}

template <typename T>
void cpt_to_pow2p1(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                   int row_stride, int col_stride, T *dcv, int lddcv, T *dv,
                   int lddv, int queue_idx) {

  int total_thread_x = ceil((float)nrow / row_stride);
  int total_thread_y = ceil((float)ncol / col_stride);
  int tbx = min(handle.B, total_thread_x);
  int tby = min(handle.B, total_thread_y);
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _cpt_to_pow2p1<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, row_stride, col_stride, dcv, lddcv, dv, lddv);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void cpt_to_pow2p1<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int ncol, int row_stride, int col_stride,
                                    double *dcv, int lddcv, double *dv,
                                    int lddv, int queue_idx);
template void cpt_to_pow2p1<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int ncol, int row_stride, int col_stride,
                                   float *dcv, int lddcv, float *dv, int lddv,
                                   int queue_idx);

/* 2D compact to (2^k)+1 with add */
template <typename T>
__global__ void _cpt_to_pow2p1_add(int nrow, int ncol, int row_stride1,
                                   int col_stride1, int row_stride2,
                                   int col_stride2, T *dcv, int lddcv, T *dv,
                                   int lddv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  for (int y = y0; y * row_stride2 < nrow; y += blockDim.y * gridDim.y) {
    for (int x = x0; x * col_stride2 < ncol; x += blockDim.x * gridDim.x) {
      int x_strided1 = x * col_stride1;
      int y_strided1 = y * row_stride1;
      int x_strided2 = x * col_stride2;
      int y_strided2 = y * row_stride2;
      dv[get_idx(lddv, y_strided2, x_strided2)] +=
          dcv[get_idx(lddcv, y_strided1, x_strided1)];
    }
  }
}

template <typename T>
void cpt_to_pow2p1_add(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                       int row_stride1, int col_stride1, int row_stride2,
                       int col_stride2, T *dcv, int lddcv, T *dv, int lddv,
                       int queue_idx) {

  int total_thread_x = ceil((float)nrow / row_stride2);
  int total_thread_y = ceil((float)ncol / col_stride2);
  int tbx = min(handle.B, total_thread_x);
  int tby = min(handle.B, total_thread_y);
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _cpt_to_pow2p1_add<<<blockPerGrid, threadsPerBlock, 0,
                       *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, row_stride1, col_stride1, row_stride2, col_stride2, dcv,
      lddcv, dv, lddv);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void cpt_to_pow2p1_add<double>(mgard_cuda_handle<double> &handle,
                                        int nrow, int ncol, int row_stride1,
                                        int col_stride1, int row_stride2,
                                        int col_stride2, double *dcv, int lddcv,
                                        double *dv, int lddv, int queue_idx);
template void cpt_to_pow2p1_add<float>(mgard_cuda_handle<float> &handle,
                                       int nrow, int ncol, int row_stride1,
                                       int col_stride1, int row_stride2,
                                       int col_stride2, float *dcv, int lddcv,
                                       float *dv, int lddv, int queue_idx);

/* 2D compact to (2^k)+1 with add */
template <typename T>
__global__ void _cpt_to_pow2p1_subtract(int nrow, int ncol, int row_stride1,
                                        int col_stride1, int row_stride2,
                                        int col_stride2, T *dcv, int lddcv,
                                        T *dv, int lddv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  for (int y = y0; y * row_stride2 < nrow; y += blockDim.y * gridDim.y) {
    for (int x = x0; x * col_stride2 < ncol; x += blockDim.x * gridDim.x) {
      int x_strided1 = x * col_stride1;
      int y_strided1 = y * row_stride1;
      int x_strided2 = x * col_stride2;
      int y_strided2 = y * row_stride2;
      dv[get_idx(lddv, y_strided2, x_strided2)] -=
          dcv[get_idx(lddcv, y_strided1, x_strided1)];
    }
  }
}

template <typename T>
void cpt_to_pow2p1_subtract(mgard_cuda_handle<T> &handle, int nrow, int ncol,
                            int row_stride1, int col_stride1, int row_stride2,
                            int col_stride2, T *dcv, int lddcv, T *dv, int lddv,
                            int queue_idx) {

  int total_thread_x = ceil((float)nrow / row_stride2);
  int total_thread_y = ceil((float)ncol / col_stride2);
  int tbx = min(handle.B, total_thread_x);
  int tby = min(handle.B, total_thread_y);
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _cpt_to_pow2p1_subtract<<<blockPerGrid, threadsPerBlock, 0,
                            *(cudaStream_t *)handle.get(queue_idx)>>>(
      nrow, ncol, row_stride1, col_stride1, row_stride2, col_stride2, dcv,
      lddcv, dv, lddv);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void cpt_to_pow2p1_subtract<double>(mgard_cuda_handle<double> &handle,
                                             int nrow, int ncol,
                                             int row_stride1, int col_stride1,
                                             int row_stride2, int col_stride2,
                                             double *dcv, int lddcv, double *dv,
                                             int lddv, int queue_idx);
template void cpt_to_pow2p1_subtract<float>(mgard_cuda_handle<float> &handle,
                                            int nrow, int ncol, int row_stride1,
                                            int col_stride1, int row_stride2,
                                            int col_stride2, float *dcv,
                                            int lddcv, float *dv, int lddv,
                                            int queue_idx);

/* 1D compact to (2^k)+1*/
template <typename T>
__global__ void _cpt_to_pow2p1(int nrow, int row_stride, T *dcv, T *dv) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  for (int x = x0; x * row_stride < nrow; x += blockDim.x * gridDim.x) {
    int x_strided = x * row_stride;
    dv[x_strided] = dcv[x];
  }
}

template <typename T>
void cpt_to_pow2p1(mgard_cuda_handle<T> &handle, int nrow, int row_stride,
                   T *dcv, T *dv, int B, int queue_idx) {

  int total_thread_x = ceil((float)nrow / row_stride);
  int total_thread_y = 1;
  int tbx = min(B, total_thread_x);
  int tby = 1;
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  _cpt_to_pow2p1<<<blockPerGrid, threadsPerBlock, 0,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(nrow, row_stride,
                                                             dcv, dv);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void cpt_to_pow2p1<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int row_stride, double *dcv, double *dv,
                                    int B, int queue_idx);
template void cpt_to_pow2p1<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int row_stride, float *dcv, float *dv, int B,
                                   int queue_idx);

template <typename T> __device__ T _dist(T *dcoord, int x, int y) {
  return dcoord[y] - dcoord[x];
}

template <typename T>
__global__ void _calc_cpt_dist(int n, int stride, T *dcoord, T *ddist) {

  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);
  T *sm = SharedMemory<T>();
  // extern __shared__ double sm[]; //size = blockDim.x + 1

  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int x0_sm = threadIdx.x;
  T dist;
  for (int x = x0; x * stride < n - 1; x += blockDim.x * gridDim.x) {
    // Load coordinates
    sm[x0_sm] = dcoord[x * stride];
    // printf("sm[%d] block %d thread %d load[%d] %f\n", x0_sm, blockIdx.x,
    // threadIdx.x, x, dcoord[x * stride]);
    if (x0_sm == 0) {
      // sm[blockDim.x] = dcoord[(x + blockDim.x) * stride];
      int left = (n - 1) / stride + 1 - blockIdx.x * blockDim.x;
      sm[min(blockDim.x, left - 1)] =
          dcoord[min((x + blockDim.x) * stride, n - 1)];
      // printf("sm[%d] extra block %d thread %d load[%d] %f\n", min(blockDim.x,
      // left-1), blockIdx.x, threadIdx.x, min((x + blockDim.x) * stride, n-1),
      // dcoord[min((x + blockDim.x) * stride, n-1)]);
    }
    __syncthreads();

    // Compute distance
    dist = _get_dist(sm, x0_sm, x0_sm + 1);
    __syncthreads();
    ddist[x] = dist;
    __syncthreads();
  }
}

template <typename T>
void calc_cpt_dist(mgard_cuda_handle<T> &handle, int nrow, int row_stride,
                   T *dcoord, T *ddist, int queue_idx) {

  int total_thread_x = ceil((float)nrow / row_stride) - 1;
  int total_thread_y = 1;
  int tbx = min(handle.B, total_thread_x);
  int tby = 1;
  int gridx = ceil((float)total_thread_x / tbx);
  int gridy = ceil((float)total_thread_y / tby);
  dim3 threadsPerBlock(tbx, tby);
  dim3 blockPerGrid(gridx, gridy);
  size_t sm_size = (tbx + 1) * sizeof(T);
  _calc_cpt_dist<<<blockPerGrid, threadsPerBlock, sm_size,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(nrow, row_stride,
                                                             dcoord, ddist);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template void calc_cpt_dist<double>(mgard_cuda_handle<double> &handle, int nrow,
                                    int row_stride, double *dcoord,
                                    double *ddist, int queue_idx);
template void calc_cpt_dist<float>(mgard_cuda_handle<float> &handle, int nrow,
                                   int row_stride, float *dcoord, float *ddist,
                                   int queue_idx);
} // namespace mgard_cuda