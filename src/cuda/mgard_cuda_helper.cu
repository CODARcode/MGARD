/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_handle.h"
#include "cuda/mgard_cuda_helper.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <sstream>   // std::stringstream
#include <stdexcept> // std::runtime_error
#include <string>
#include <utility> // std::pair
#include <vector>

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"

namespace mgard_cuda {

// print 2D CPU
template <typename T> void print_matrix(int nrow, int ncol, T *v, int ldv) {
  // std::cout << std::setw(10);
  // std::cout << std::setprecision(2) << std::fixed;
  // for (int i = 0; i < nrow; i++) {
  //   for (int j = 0; j < ncol; j++) {
  //     if (isnan(v[ldv * i + j])) {
  //       std::cout << "nan\n";
  //       return;
  //     }
  //     if (isinf(v[ldv * i + j])) {
  //       std::cout << "inf\n";
  //       return;
  //     }
  //     if (abs(v[ldv * i + j]) > 10000) {
  //       std::cout << "LARGE" << " (" << i << ", " << j << ")\n";
  //       return;
  //     }
  //     // if ((int)(v[ldv * i + j]*10)%10 != 0) {
  //     //   std::cout << "NOT INTEGER" << " (" << i << ", " << j << ")\n";
  //     //   return;
  //     // }
  //   }
  // }

  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      std::cout << std::setw(5) << std::setprecision(3) << std::fixed
                << v[ldv * i + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// print 2D GPU
template <typename T>
void print_matrix_cuda(int nrow, int ncol, T *dv, int lddv) {
  // std::cout << std::setw(10);
  // std::cout << std::setprecision(2) << std::fixed;
  mgard_cuda_handle<float, 2> *tmp_handle = new mgard_cuda_handle<float, 2>();
  int queue_idx = 0;
  T *v = new T[nrow * ncol];
  cudaMemcpy2DAsyncHelper(*tmp_handle, v, ncol * sizeof(T), dv,
                          lddv * sizeof(T), ncol * sizeof(T), nrow, D2H,
                          queue_idx);
  tmp_handle->sync(queue_idx);
  print_matrix(nrow, ncol, v, ncol);
  delete[] v;
  delete tmp_handle;
}

// print 3D GPU
template <typename T>
void print_matrix_cuda(int nrow, int ncol, int nfib, T *dv, int lddv1,
                       int lddv2, int sizex) {
  std::cout << std::setw(10);
  std::cout << std::setprecision(2) << std::fixed;
  mgard_cuda_handle<float, 3> *tmp_handle = new mgard_cuda_handle<float, 3>();
  int queue_idx = 0;

  T *v = new T[nrow * ncol * nfib];
  cudaMemcpy3DAsyncHelper(*tmp_handle, v, nfib * sizeof(T), nfib * sizeof(T),
                          ncol, dv, lddv1 * sizeof(T), sizex * sizeof(T), lddv2,
                          nfib * sizeof(T), ncol, nrow, D2H, queue_idx);
  tmp_handle->sync(queue_idx);
  print_matrix(nrow, ncol, nfib, v, nfib, ncol);
  delete[] v;
  delete tmp_handle;
}

// print 3D CPU
template <typename T>
void print_matrix(int nrow, int ncol, int nfib, T *v, int ldv1, int ldv2) {
  std::cout << std::setw(10);
  std::cout << std::setprecision(2) << std::fixed;
  for (int i = 0; i < nrow; i++) {
    std::cout << "[ nrow = " << i << " ]\n";
    print_matrix(ncol, nfib, v + i * ldv1 * ldv2, ldv1);
    // std::cout << std::endl;
  }
}

// compare 2D CPU
template <typename T>
bool compare_matrix(int nrow, int ncol, T *v1, int ldv1, T *v2, int ldv2) {
  // std::cout << std::setw(10);
  // std::cout << std::setprecision(2) << std::fixed;
  bool correct = true;
  bool nan = false;
  double E = 1e-6;
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      T a = v1[ldv1 * i + j];
      T b = v2[ldv2 * i + j];
      double diff = a - b;
      diff = abs(diff);
      if (diff > E) {
        correct = false;
        std::cout << "Diff at (" << i << ", " << j << ") ";
        std::cout << a << " - " << b << " = " << diff << std::endl;
      }
      if (isnan(a) || isnan(b)) {
        nan = true;
        // std::cout << "NAN at (" << i << ", " << j << ") ";
        // std::cout << a << " - " << b << " = " << abs(a-b) << std::endl;
      }
    }
  }
  if (correct)
    printf("Compare: correct.\n");
  else
    printf("Compare: wrong.\n");
  if (nan)
    printf("Nan: include.\n");
  // else printf("Nan: not include.\n");
  return correct;
}

// compare 2D GPU
template <typename T>
bool compare_matrix_cuda(int nrow, int ncol, T *dv1, int lddv1, T *dv2,
                         int lddv2) {
  mgard_cuda_handle<float, 2> *tmp_handle = new mgard_cuda_handle<float, 2>();
  int queue_idx = 0;

  T *v1 = new T[nrow * ncol];
  int ldv1 = ncol;
  cudaMemcpy2DAsyncHelper(*tmp_handle, v1, ldv1 * sizeof(T), dv1,
                          lddv1 * sizeof(T), ncol * sizeof(T), nrow, D2H,
                          queue_idx);
  T *v2 = new T[nrow * ncol];
  int ldv2 = ncol;
  cudaMemcpy2DAsyncHelper(*tmp_handle, v2, ldv2 * sizeof(T), dv2,
                          lddv2 * sizeof(T), ncol * sizeof(T), nrow, D2H,
                          queue_idx);
  tmp_handle->sync(queue_idx);
  bool ret = compare_matrix(nrow, ncol, v1, ldv1, v2, ldv2);
  delete[] v1;
  delete[] v2;
  delete tmp_handle;
  return ret;
}

// compare 3D CPU
template <typename T>
bool compare_matrix(int nrow, int ncol, int nfib, T *v1, int ldv11, int ldv12,
                    T *v2, int ldv21, int ldv22, bool print_matrix) {
  // std::cout << std::setw(10);
  // std::cout << std::setprecision(2) << std::fixed;
  bool correct = true;
  bool nan = false;
  double E = 1e-5;
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      for (int k = 0; k < nfib; k++) {
        T a = v1[ldv11 * ldv12 * i + ldv11 * j + k];
        T b = v2[ldv21 * ldv22 * i + ldv21 * j + k];
        double diff = a - b;
        diff = abs(diff);
        if (diff > E) {
          correct = false;
          // std::cout << "Diff at (" << i << ", " << j << ", " << k <<") ";
          // std::cout << a << " - " << b << " = " << abs(a-b) << std::endl;
          if (print_matrix)
            std::cout << ANSI_RED << std::setw(9) << std::setprecision(6)
                      << std::fixed << b << ", " << ANSI_RESET;
          //<< b << "(" << a << ")"<< ", " << ANSI_RESET;
        } else {
          if (isnan(b)) {
            if (print_matrix)
              std::cout << ANSI_RED << std::setw(9) << std::setprecision(6)
                        << std::fixed << b << ", " << ANSI_RESET;
          } else {
            if (print_matrix)
              std::cout << ANSI_GREEN << std::setw(9) << std::setprecision(6)
                        << std::fixed << b << ", " << ANSI_RESET;
          }
        }

        if (std::isnan(a) || std::isnan(b)) {
          nan = true;
          // std::cout << "NAN at (" << i << ", " << j << ") ";
          // std::cout << a << " - " << b << " = " << abs(a-b) << std::endl;
        }
      }
      if (print_matrix)
        std::cout << std::endl;
    }
    if (print_matrix)
      std::cout << std::endl;
  }
  if (correct && !nan)
    printf(ANSI_GREEN "Compare: correct.\n" ANSI_RESET);
  else
    printf(ANSI_RED "Compare: wrong.\n" ANSI_RESET);
  if (nan)
    printf(ANSI_RED "Nan: include.\n" ANSI_RESET);
  // else printf("Nan: not include.\n");
  return correct;
}

// compare 3D GPU
template <typename T>
bool compare_matrix_cuda(int nrow, int ncol, int nfib, T *dv1, int lddv11,
                         int lddv12, int sizex1, T *dv2, int lddv21, int lddv22,
                         int sizex2, bool print_matrix) {
  mgard_cuda_handle<float, 3> *tmp_handle = new mgard_cuda_handle<float, 3>();
  int queue_idx = 0;

  T *v1 = new T[nrow * ncol * nfib];
  int ldv11 = nfib;
  int ldv12 = ncol;
  cudaMemcpy3DAsyncHelper(*tmp_handle, v1, ldv11 * sizeof(T), nfib * sizeof(T),
                          ldv12, dv1, lddv11 * sizeof(T), sizex1 * sizeof(T),
                          lddv12, nfib * sizeof(T), ncol, nrow, D2H, queue_idx);

  T *v2 = new T[nrow * ncol * nfib];
  int ldv21 = nfib;
  int ldv22 = ncol;
  cudaMemcpy3DAsyncHelper(*tmp_handle, v2, ldv21 * sizeof(T), nfib * sizeof(T),
                          ldv22, dv2, lddv21 * sizeof(T), sizex2 * sizeof(T),
                          lddv22, nfib * sizeof(T), ncol, nrow, D2H, queue_idx);
  tmp_handle->sync(queue_idx);
  bool ret = compare_matrix(nrow, ncol, nfib, v1, ldv11, ldv12, v2, ldv21,
                            ldv22, print_matrix);
  delete[] v1;
  delete[] v2;
  delete tmp_handle;
  return ret;
}

// print 3D CPU
template <typename T>
void verify_matrix(int nrow, int ncol, int nfib, T *v, int ldv1, int ldv2,
                   std::string file_prefix, bool store, bool verify) {
  std::string filename = file_prefix + ".csv";
  if (store) {
    std::ofstream myfile;
    myfile.open(filename, std::ios::out | std::ios::binary);
    if (!myfile) {
      printf("Error: cannot write file\n");
      return;
    }
    myfile.write((char *)v, nrow * ncol * nfib * sizeof(T));
    myfile.close();
    if (!myfile.good()) {
      printf("Error occurred at write time!\n");
      return;
    }
  }
  if (verify) {
    std::fstream fin;
    fin.open(filename, std::ios::in | std::ios::binary);
    if (!fin) {
      printf("Error: cannot read file\n");
      return;
    }
    T *v2 = new T[nrow * ncol * nfib];
    fin.read((char *)v2, nrow * ncol * nfib * sizeof(T));
    fin.close();
    if (!fin.good()) {
      printf("Error occurred at reading time!\n");
      return;
    }

    bool mismatch = false;
    for (int i = 0; i < nrow; i++) {
      for (int j = 0; j < ncol; j++) {
        for (int k = 0; k < nfib; k++) {
          if (v[get_idx(ldv1, ldv2, i, j, k)] !=
              v2[get_idx(nfib, ncol, i, j, k)]) {
            std::cout << filename << ": ";
            printf("Mismatch[%d %d %d] %f - %f\n", i, j, k,
                   v[get_idx(ldv1, ldv2, i, j, k)],
                   v2[get_idx(nfib, ncol, i, j, k)]);
            mismatch = true;
          }
        }
      }
    }

    delete v2;
    if (mismatch)
      exit(-1);
  }
}

// print 3D GPU
template <typename T>
void verify_matrix_cuda(int nrow, int ncol, int nfib, T *dv, int lddv1,
                        int lddv2, int sizex, std::string file_prefix,
                        bool store, bool verify) {
  // std::cout << std::setw(10);
  // std::cout << std::setprecision(2) << std::fixed;
  if (store || verify) {
    mgard_cuda_handle<float, 3> *tmp_handle = new mgard_cuda_handle<float, 3>();
    int queue_idx = 0;

    T *v = new T[nrow * ncol * nfib];
    cudaMemcpy3DAsyncHelper(*tmp_handle, v, nfib * sizeof(T), nfib * sizeof(T),
                            ncol, dv, lddv1 * sizeof(T), sizex * sizeof(T),
                            lddv2, nfib * sizeof(T), ncol, nrow, D2H,
                            queue_idx);
    tmp_handle->sync(queue_idx);
    verify_matrix(nrow, ncol, nfib, v, nfib, ncol, file_prefix, store, verify);
    delete[] v;
    delete tmp_handle;
  }
}

// Allocate 1D
void cudaMallocHelper(void **devPtr, size_t size) {
  gpuErrchk(cudaMalloc(devPtr, size));
}

// Allocate 2D
void cudaMallocPitchHelper(void **devPtr, size_t *pitch, size_t width,
                           size_t height) {
  gpuErrchk(cudaMallocPitch(devPtr, pitch, width, height));
}

// Allocate 3D
void cudaMalloc3DHelper(void **devPtr, size_t *pitch, size_t width,
                        size_t height, size_t depth) {
  cudaPitchedPtr devPitchedPtr;
  cudaExtent extent = make_cudaExtent(width, height, depth);
  gpuErrchk(cudaMalloc3D(&devPitchedPtr, extent));
  *devPtr = devPitchedPtr.ptr;
  *pitch = devPitchedPtr.pitch;
}
// Allocate page-locked memory on host
void cudaMallocHostHelper(void **ptr, size_t size) {
  gpuErrchk(cudaMallocHost(ptr, size));
}

// Copy 1D
template <typename T, int D>
void cudaMemcpyAsyncHelper(mgard_cuda_handle<T, D> &handle, void *dst,
                           const void *src, size_t count, enum copy_type kind,
                           int queue_idx) {

  // printf("copu: %llu\n", count);
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);
  enum cudaMemcpyKind cuda_copy_type;
  switch (kind) {
  case H2D:
    cuda_copy_type = cudaMemcpyHostToDevice;
    break;
  case D2H:
    cuda_copy_type = cudaMemcpyDeviceToHost;
    break;
  case D2D:
    cuda_copy_type = cudaMemcpyDeviceToDevice;
    break;
  }
  gpuErrchk(cudaMemcpyAsync(dst, src, count, cuda_copy_type, stream));
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

// Copy 2D
template <typename T, int D>
void cudaMemcpy2DAsyncHelper(mgard_cuda_handle<T, D> &handle, void *dst,
                             size_t dpitch, void *src, size_t spitch,
                             size_t width, size_t height, enum copy_type kind,
                             int queue_idx) {

  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);
  enum cudaMemcpyKind cuda_copy_type;
  switch (kind) {
  case H2D:
    cuda_copy_type = cudaMemcpyHostToDevice;
    break;
  case D2H:
    cuda_copy_type = cudaMemcpyDeviceToHost;
    break;
  case D2D:
    cuda_copy_type = cudaMemcpyDeviceToDevice;
    break;
  }
  gpuErrchk(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                              cuda_copy_type, stream));
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

// Copy 3D
template <typename T, int D>
void cudaMemcpy3DAsyncHelper(mgard_cuda_handle<T, D> &handle, void *dst,
                             size_t dpitch, size_t dwidth, size_t dheight,
                             void *src, size_t spitch, size_t swidth,
                             size_t sheight, size_t width, size_t height,
                             size_t depth, enum copy_type kind, int queue_idx) {

  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  cudaExtent extent = make_cudaExtent(width, height, depth);
  cudaMemcpy3DParms p = {0};
  p.dstPtr.ptr = dst;
  p.dstPtr.pitch = dpitch;
  p.dstPtr.xsize = dwidth;
  p.dstPtr.ysize = dheight;

  p.srcPtr.ptr = src;
  p.srcPtr.pitch = spitch;
  p.srcPtr.xsize = swidth;
  p.srcPtr.ysize = sheight;

  p.extent = extent;
  enum cudaMemcpyKind cuda_copy_type;
  switch (kind) {
  case H2D:
    cuda_copy_type = cudaMemcpyHostToDevice;
    break;
  case D2H:
    cuda_copy_type = cudaMemcpyDeviceToHost;
    break;
  case D2D:
    cuda_copy_type = cudaMemcpyDeviceToDevice;
    break;
  }
  p.kind = cuda_copy_type;

  gpuErrchk(cudaMemcpy3DAsync(&p, stream));
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

void cudaFreeHelper(void *devPtr) { gpuErrchk(cudaFree(devPtr)); }

void cudaFreeHostHelper(void *ptr) { gpuErrchk(cudaFreeHost(ptr)); }

void cudaMemsetHelper(void *devPtr, int value, size_t count) {
  gpuErrchk(cudaMemset(devPtr, value, count));
}

void cudaMemset2DHelper(void *devPtr, size_t pitch, int value, size_t width,
                        size_t height) {
  gpuErrchk(cudaMemset2D(devPtr, pitch, value, width, height));
}

void cudaMemset3DHelper(void *devPtr, size_t pitch, size_t dwidth,
                        size_t dheight, int value, size_t width, size_t height,
                        size_t depth) {
  cudaExtent extent = make_cudaExtent(width, height, depth);
  cudaPitchedPtr devPitchedPtr;
  devPitchedPtr.ptr = devPtr;
  devPitchedPtr.pitch = pitch;
  devPitchedPtr.xsize = dwidth;
  devPitchedPtr.ysize = dheight;
  gpuErrchk(cudaMemset3D(devPitchedPtr, value, extent));
}

void cudaSetDeviceHelper(int dev_id) { gpuErrchk(cudaSetDevice(dev_id)); }

// Copy 1D Peer
template <typename T, int D>
void cudaMemcpyPeerAsyncHelper(mgard_cuda_handle<T, D> &handle, void *dst,
                               int dst_dev, const void *src, int src_dev,
                               size_t count, int queue_idx) {

  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);
  gpuErrchk(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, count, stream));
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

// Copy 3D peer
template <typename T, int D>
void cudaMemcpy3DPeerAsyncHelper(mgard_cuda_handle<T, D> &handle, void *dst,
                                 int dst_dev, size_t dpitch, size_t dwidth,
                                 size_t dheight, void *src, int src_dev,
                                 size_t spitch, size_t swidth, size_t sheight,
                                 size_t width, size_t height, size_t depth,
                                 int queue_idx) {

  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  cudaExtent extent = make_cudaExtent(width, height, depth);
  cudaMemcpy3DPeerParms p = {0};
  p.dstPtr.ptr = dst;
  p.dstPtr.pitch = dpitch;
  p.dstPtr.xsize = dwidth;
  p.dstPtr.ysize = dheight;

  p.srcPtr.ptr = src;
  p.srcPtr.pitch = spitch;
  p.srcPtr.xsize = swidth;
  p.srcPtr.ysize = sheight;

  p.extent = extent;

  p.dstDevice = dst_dev;
  p.srcDevice = src_dev;

  // printf("src_dev: %d - dst_dev: %d\n", dst_dev, src_dev);

  gpuErrchk(cudaMemcpy3DPeerAsync(&p, stream));
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

bool isGPUPointer(void *ptr) {
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, ptr);
  return attr.type == cudaMemoryTypeDevice;
}

#define KERNELS(T)                                                             \
  template bool compare_matrix<T>(int nrow, int ncol, T *v1, int ldv1, T *v2,  \
                                  int ldv2);                                   \
  template bool compare_matrix_cuda<T>(int nrow, int ncol, T *dv1, int lddv1,  \
                                       T *dv2, int lddv2);                     \
  template bool compare_matrix<T>(int nrow, int ncol, int nfib, T *v1,         \
                                  int ldv11, int ldv12, T *v2, int ldv21,      \
                                  int ldv22, bool print_matrix);               \
  template bool compare_matrix_cuda<T>(int nrow, int ncol, int nfib, T *dv1,   \
                                       int lddv11, int lddv12, int sizex1,     \
                                       T *dv2, int lddv21, int lddv22,         \
                                       int sizex2, bool print_matrix);         \
  template void verify_matrix<T>(int nrow, int ncol, int nfib, T *v, int ldv1, \
                                 int ldv2, std::string file_prefix, bool save, \
                                 bool verify);                                 \
  template void verify_matrix_cuda<T>(                                         \
      int nrow, int ncol, int nfib, T *dv, int lddv1, int lddv2, int sizex,    \
      std::string file_prefix, bool save, bool verify);

KERNELS(double)
KERNELS(float)
KERNELS(int)
KERNELS(unsigned int)
KERNELS(size_t)
KERNELS(uint8_t)
#undef KERNELS

#define KERNELS(T)                                                             \
  template void print_matrix<T>(int nrow, int ncol, T *v, int ldv);            \
  template void print_matrix_cuda<T>(int nrow, int ncol, T *dv, int lddv);     \
  template void print_matrix_cuda<T>(int nrow, int ncol, int nfib, T *dv,      \
                                     int lddv1, int lddv2, int sizex);         \
  template void print_matrix<T>(int nrow, int ncol, int nfib, T *v, int ldv1,  \
                                int ldv2);

KERNELS(double)
KERNELS(float)
KERNELS(int)
KERNELS(unsigned int)
KERNELS(unsigned long)
#undef KERNELS

#define KERNELS(T, D)                                                          \
  template void cudaMemcpyAsyncHelper<T, D>(                                   \
      mgard_cuda_handle<T, D> & handle, void *dst, const void *src,            \
      size_t count, enum copy_type kind, int queue_idx);                       \
  template void cudaMemcpy2DAsyncHelper<T, D>(                                 \
      mgard_cuda_handle<T, D> & handle, void *dst, size_t dpitch, void *src,   \
      size_t spitch, size_t width, size_t height, enum copy_type kind,         \
      int queue_idx);                                                          \
  template void cudaMemcpy3DAsyncHelper<T, D>(                                 \
      mgard_cuda_handle<T, D> & handle, void *dst, size_t dpitch,              \
      size_t dwidth, size_t dheight, void *src, size_t spitch, size_t swidth,  \
      size_t sheight, size_t width, size_t height, size_t depth,               \
      enum copy_type kind, int queue_idx);                                     \
  template void cudaMemcpyPeerAsyncHelper<T, D>(                               \
      mgard_cuda_handle<T, D> & handle, void *dst, int dst_dev,                \
      const void *src, int src_dev, size_t count, int queue_idx);              \
  template void cudaMemcpy3DPeerAsyncHelper<T, D>(                             \
      mgard_cuda_handle<T, D> & handle, void *dst, int dst_dev, size_t dpitch, \
      size_t dwidth, size_t dheight, void *src, int src_dev, size_t spitch,    \
      size_t swidth, size_t sheight, size_t width, size_t height,              \
      size_t depth, int queue_idx);

KERNELS(double, 1)
KERNELS(float, 1)
KERNELS(double, 2)
KERNELS(float, 2)
KERNELS(double, 3)
KERNELS(float, 3)
KERNELS(double, 4)
KERNELS(float, 4)
KERNELS(double, 5)
KERNELS(float, 5)
#undef KERNELS
} // namespace mgard_cuda