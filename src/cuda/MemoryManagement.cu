/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

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

#include "cuda/CommonInternal.h"

#include "cuda/Handle.h"
#include "cuda/MemoryManagement.h"

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"

namespace mgard_cuda {

template <typename SubArrayType>
void PrintSubarray(std::string name, SubArrayType subArray) {
  Handle<1, float> tmp_handle;

  SIZE nrow = 1;
  SIZE ncol = 1;
  SIZE nfib = 1;

  nfib = subArray.shape[0];
  ncol = subArray.shape[1];
  nrow = subArray.shape[2];

  using T = typename SubArrayType::DataType;
  T *v = new T[nrow * ncol * nfib];
  cudaMemcpy3DAsyncHelper(tmp_handle, v, nfib * sizeof(T), nfib * sizeof(T),
                          ncol, subArray.data(), subArray.lddv1 * sizeof(T),
                          nfib * sizeof(T), subArray.lddv2, nfib * sizeof(T),
                          ncol, nrow, D2H, 0);
  tmp_handle.sync(0);

  std::cout << "SubArray: " << name << "(" << nrow << " * " << ncol << " * "
            << nfib << ") sizeof(T) = " << sizeof(T) << std::endl;
  for (int i = 0; i < nrow; i++) {
    printf("[i = %d]\n", i);
    for (int j = 0; j < ncol; j++) {
      for (int k = 0; k < nfib; k++) {
        if (std::is_same<T, std::uint8_t>::value) {
          std::cout << std::setw(8)
                    << (unsigned int)v[nfib * ncol * i + nfib * j + k] << ", ";
        } else {
          std::cout << std::setw(8) << std::setprecision(6) << std::fixed
                    << v[nfib * ncol * i + nfib * j + k] << ", ";
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  delete[] v;
}

// print 2D CPU
template <typename T> void print_matrix(SIZE nrow, SIZE ncol, T *v, SIZE ldv) {
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
      std::cout << std::setw(8) << std::setprecision(6) << std::fixed
                << v[ldv * i + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// print 2D GPU
template <typename T>
void print_matrix_cuda(SIZE nrow, SIZE ncol, T *dv, SIZE lddv) {
  // std::cout << std::setw(10);
  // std::cout << std::setprecision(2) << std::fixed;
  Handle<2, float> *tmp_handle = new Handle<2, float>();
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
void print_matrix_cuda(SIZE nrow, SIZE ncol, SIZE nfib, T *dv, SIZE lddv1,
                       SIZE lddv2, SIZE sizex) {
  std::cout << std::setw(10);
  std::cout << std::setprecision(2) << std::fixed;
  Handle<3, float> *tmp_handle = new Handle<3, float>();
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
void print_matrix(SIZE nrow, SIZE ncol, SIZE nfib, T *v, SIZE ldv1, SIZE ldv2) {
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
bool compare_matrix(SIZE nrow, SIZE ncol, T *v1, SIZE ldv1, T *v2, SIZE ldv2) {
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
bool compare_matrix_cuda(SIZE nrow, SIZE ncol, T *dv1, SIZE lddv1, T *dv2,
                         SIZE lddv2) {
  Handle<2, float> *tmp_handle = new Handle<2, float>();
  int queue_idx = 0;

  T *v1 = new T[nrow * ncol];
  SIZE ldv1 = ncol;
  cudaMemcpy2DAsyncHelper(*tmp_handle, v1, ldv1 * sizeof(T), dv1,
                          lddv1 * sizeof(T), ncol * sizeof(T), nrow, D2H,
                          queue_idx);
  T *v2 = new T[nrow * ncol];
  SIZE ldv2 = ncol;
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
bool compare_matrix(SIZE nrow, SIZE ncol, SIZE nfib, T *v1, SIZE ldv11,
                    SIZE ldv12, T *v2, SIZE ldv21, SIZE ldv22,
                    bool print_matrix) {
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
bool compare_matrix_cuda(SIZE nrow, SIZE ncol, SIZE nfib, T *dv1, SIZE lddv11,
                         SIZE lddv12, SIZE sizex1, T *dv2, SIZE lddv21,
                         SIZE lddv22, SIZE sizex2, bool print_matrix) {
  Handle<3, float> *tmp_handle = new Handle<3, float>();
  int queue_idx = 0;

  T *v1 = new T[nrow * ncol * nfib];
  SIZE ldv11 = nfib;
  SIZE ldv12 = ncol;
  cudaMemcpy3DAsyncHelper(*tmp_handle, v1, ldv11 * sizeof(T), nfib * sizeof(T),
                          ldv12, dv1, lddv11 * sizeof(T), sizex1 * sizeof(T),
                          lddv12, nfib * sizeof(T), ncol, nrow, D2H, queue_idx);

  T *v2 = new T[nrow * ncol * nfib];
  SIZE ldv21 = nfib;
  SIZE ldv22 = ncol;
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
void verify_matrix(SIZE nrow, SIZE ncol, SIZE nfib, T *v, SIZE ldv1, SIZE ldv2,
                   std::string file_prefix, bool store, bool verify) {
  std::string filename = file_prefix + ".dat";
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
void verify_matrix_cuda(SIZE nrow, SIZE ncol, SIZE nfib, T *dv, SIZE lddv1,
                        SIZE lddv2, SIZE sizex, std::string file_prefix,
                        bool store, bool verify) {
  // std::cout << std::setw(10);
  // std::cout << std::setprecision(2) << std::fixed;
  if (store || verify) {
    Handle<3, float> *tmp_handle = new Handle<3, float>();
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
template <DIM D, typename T>
void cudaMallocHelper(Handle<D, T> &handle, void **devPtr, size_t size) {
  gpuErrchk(cudaMalloc(devPtr, size));
}

// Allocate 2D
template <DIM D, typename T>
void cudaMallocPitchHelper(Handle<D, T> &handle, void **devPtr, size_t *pitch,
                           size_t width, size_t height) {
  if (handle.reduce_memory_footprint) {
    cudaMallocHelper(handle, devPtr, width * height);
    *pitch = width;
  } else {
    gpuErrchk(cudaMallocPitch(devPtr, pitch, width, height));
  }
}

// Allocate 3D
template <DIM D, typename T>
void cudaMalloc3DHelper(Handle<D, T> &handle, void **devPtr, size_t *pitch,
                        size_t width, size_t height, size_t depth) {

  if (handle.reduce_memory_footprint) {
    cudaMallocHelper(handle, devPtr, width * height * depth);
    *pitch = width;
  } else {
    cudaPitchedPtr devPitchedPtr;
    cudaExtent extent = make_cudaExtent(width, height, depth);
    gpuErrchk(cudaMalloc3D(&devPitchedPtr, extent));
    *devPtr = devPitchedPtr.ptr;
    *pitch = devPitchedPtr.pitch;
  }
}

// Allocate page-locked memory on host
void cudaMallocHostHelper(void **ptr, size_t size) {
  gpuErrchk(cudaMallocHost(ptr, size));
}

enum cudaMemcpyKind inferTransferType(enum copy_type kind) {
  switch (kind) {
  case H2D:
    return cudaMemcpyHostToDevice;
  case D2H:
    return cudaMemcpyDeviceToHost;
  case D2D:
    return cudaMemcpyDeviceToDevice;
  case H2H:
    return cudaMemcpyHostToHost;
  case AUTO:
    return cudaMemcpyDefault;
  }
}

// Copy 1D
template <DIM D, typename T>
void cudaMemcpyAsyncHelper(Handle<D, T> &handle, void *dst, const void *src,
                           size_t count, enum copy_type kind, int queue_idx) {

  // printf("copu: %llu\n", count);
  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);
  enum cudaMemcpyKind cuda_copy_type = inferTransferType(kind);
  // switch (kind) {
  // case H2D:
  //   cuda_copy_type = cudaMemcpyHostToDevice;
  //   break;
  // case D2H:
  //   cuda_copy_type = cudaMemcpyDeviceToHost;
  //   break;
  // case D2D:
  //   cuda_copy_type = cudaMemcpyDeviceToDevice;
  //   break;
  // }
  gpuErrchk(cudaMemcpyAsync(dst, src, count, cuda_copy_type, stream));
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

// Copy 2D
template <DIM D, typename T>
void cudaMemcpy2DAsyncHelper(Handle<D, T> &handle, void *dst, size_t dpitch,
                             void *src, size_t spitch, size_t width,
                             size_t height, enum copy_type kind,
                             int queue_idx) {

  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);
  enum cudaMemcpyKind cuda_copy_type = inferTransferType(kind);
  // switch (kind) {
  // case H2D:
  //   cuda_copy_type = cudaMemcpyHostToDevice;
  //   break;
  // case D2H:
  //   cuda_copy_type = cudaMemcpyDeviceToHost;
  //   break;
  // case D2D:
  //   cuda_copy_type = cudaMemcpyDeviceToDevice;
  //   break;
  // }
  gpuErrchk(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                              cuda_copy_type, stream));
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

void *cast_to_non_const(const void *const_ptr) {
  const unsigned long long int const_address =
      (unsigned long long int)const_ptr;
  unsigned long long int address = const_address;
  return (void *)address;
}

// Copy 3D
template <DIM D, typename T>
void cudaMemcpy3DAsyncHelper(Handle<D, T> &handle, void *dst, size_t dpitch,
                             size_t dwidth, size_t dheight, const void *src,
                             size_t spitch, size_t swidth, size_t sheight,
                             size_t width, size_t height, size_t depth,
                             enum copy_type kind, int queue_idx) {

  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  cudaExtent extent = make_cudaExtent(width, height, depth);
  cudaMemcpy3DParms p = {0};
  p.dstPtr.ptr = dst;
  p.dstPtr.pitch = dpitch;
  p.dstPtr.xsize = dwidth;
  p.dstPtr.ysize = dheight;

  p.srcPtr.ptr = cast_to_non_const(src);
  p.srcPtr.pitch = spitch;
  p.srcPtr.xsize = swidth;
  p.srcPtr.ysize = sheight;

  p.extent = extent;
  enum cudaMemcpyKind cuda_copy_type = inferTransferType(kind);
  // switch (kind) {
  // case H2D:
  //   cuda_copy_type = cudaMemcpyHostToDevice;
  //   break;
  // case D2H:
  //   cuda_copy_type = cudaMemcpyDeviceToHost;
  //   break;
  // case D2D:
  //   cuda_copy_type = cudaMemcpyDeviceToDevice;
  //   break;
  // }
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
template <DIM D, typename T>
void cudaMemcpyPeerAsyncHelper(Handle<D, T> &handle, void *dst, int dst_dev,
                               const void *src, int src_dev, size_t count,
                               int queue_idx) {

  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);
  gpuErrchk(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, count, stream));
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

// Copy 3D peer
template <DIM D, typename T>
void cudaMemcpy3DPeerAsyncHelper(Handle<D, T> &handle, void *dst, int dst_dev,
                                 size_t dpitch, size_t dwidth, size_t dheight,
                                 const void *src, int src_dev, size_t spitch,
                                 size_t swidth, size_t sheight, size_t width,
                                 size_t height, size_t depth, int queue_idx) {

  cudaStream_t stream = *(cudaStream_t *)handle.get(queue_idx);

  cudaExtent extent = make_cudaExtent(width, height, depth);
  cudaMemcpy3DPeerParms p = {0};
  p.dstPtr.ptr = dst;
  p.dstPtr.pitch = dpitch;
  p.dstPtr.xsize = dwidth;
  p.dstPtr.ysize = dheight;

  p.srcPtr.ptr = cast_to_non_const(src);
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
  template bool compare_matrix<T>(SIZE nrow, SIZE ncol, T * v1, SIZE ldv1,     \
                                  T * v2, SIZE ldv2);                          \
  template bool compare_matrix_cuda<T>(SIZE nrow, SIZE ncol, T * dv1,          \
                                       SIZE lddv1, T * dv2, SIZE lddv2);       \
  template bool compare_matrix<T>(SIZE nrow, SIZE ncol, SIZE nfib, T * v1,     \
                                  SIZE ldv11, SIZE ldv12, T * v2, SIZE ldv21,  \
                                  SIZE ldv22, bool print_matrix);              \
  template bool compare_matrix_cuda<T>(                                        \
      SIZE nrow, SIZE ncol, SIZE nfib, T * dv1, SIZE lddv11, SIZE lddv12,      \
      SIZE sizex1, T * dv2, SIZE lddv21, SIZE lddv22, SIZE sizex2,             \
      bool print_matrix);                                                      \
  template void verify_matrix<T>(                                              \
      SIZE nrow, SIZE ncol, SIZE nfib, T * v, SIZE ldv1, SIZE ldv2,            \
      std::string file_prefix, bool save, bool verify);                        \
  template void verify_matrix_cuda<T>(                                         \
      SIZE nrow, SIZE ncol, SIZE nfib, T * dv, SIZE lddv1, SIZE lddv2,         \
      SIZE sizex, std::string file_prefix, bool save, bool verify);

KERNELS(double)
KERNELS(float)
KERNELS(int)
KERNELS(unsigned int)
KERNELS(size_t)
KERNELS(uint8_t)
#undef KERNELS

#define KERNELS(T)                                                             \
  template void print_matrix<T>(SIZE nrow, SIZE ncol, T * v, SIZE ldv);        \
  template void print_matrix_cuda<T>(SIZE nrow, SIZE ncol, T * dv, SIZE lddv); \
  template void print_matrix_cuda<T>(SIZE nrow, SIZE ncol, SIZE nfib, T * dv,  \
                                     SIZE lddv1, SIZE lddv2, SIZE sizex);      \
  template void print_matrix<T>(SIZE nrow, SIZE ncol, SIZE nfib, T * v,        \
                                SIZE ldv1, SIZE ldv2);

KERNELS(double)
KERNELS(float)
KERNELS(int)
KERNELS(unsigned int)
KERNELS(LENGTH)
#undef KERNELS

#define KERNELS(D, T)                                                          \
  template void cudaMallocHelper<D, T>(Handle<D, T> & handle, void **devPtr,   \
                                       size_t size);                           \
  template void cudaMallocPitchHelper<D, T>(Handle<D, T> & handle,             \
                                            void **devPtr, size_t *pitch,      \
                                            size_t width, size_t height);      \
  template void cudaMalloc3DHelper<D, T>(Handle<D, T> & handle, void **devPtr, \
                                         size_t *pitch, size_t width,          \
                                         size_t height, size_t depth);         \
  template void cudaMemcpyAsyncHelper<D, T>(                                   \
      Handle<D, T> & handle, void *dst, const void *src, size_t count,         \
      enum copy_type kind, int queue_idx);                                     \
  template void cudaMemcpy2DAsyncHelper<D, T>(                                 \
      Handle<D, T> & handle, void *dst, size_t dpitch, void *src,              \
      size_t spitch, size_t width, size_t height, enum copy_type kind,         \
      int queue_idx);                                                          \
  template void cudaMemcpy3DAsyncHelper<D, T>(                                 \
      Handle<D, T> & handle, void *dst, size_t dpitch, size_t dwidth,          \
      size_t dheight, const void *src, size_t spitch, size_t swidth,           \
      size_t sheight, size_t width, size_t height, size_t depth,               \
      enum copy_type kind, int queue_idx);                                     \
  template void cudaMemcpyPeerAsyncHelper<D, T>(                               \
      Handle<D, T> & handle, void *dst, int dst_dev, const void *src,          \
      int src_dev, size_t count, int queue_idx);                               \
  template void cudaMemcpy3DPeerAsyncHelper<D, T>(                             \
      Handle<D, T> & handle, void *dst, int dst_dev, size_t dpitch,            \
      size_t dwidth, size_t dheight, const void *src, int src_dev,             \
      size_t spitch, size_t swidth, size_t sheight, size_t width,              \
      size_t height, size_t depth, int queue_idx);

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)
#undef KERNELS

#define KERNELS(D, T)                                                          \
  template void PrintSubarray<SubArray<D, T>>(std::string name,                \
                                              SubArray<D, T> subArray);

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)
KERNELS(2, uint8_t)

#undef KERNELS

} // namespace mgard_cuda