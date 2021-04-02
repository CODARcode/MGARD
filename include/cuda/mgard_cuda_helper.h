/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGARD_CUDA_HELPER
#define MGARD_CUDA_HELPER

#include "cuda/mgard_cuda_common_internal.h"

#include <string>

namespace mgard_cuda {

enum copy_type { H2D, D2H, D2D };

template <typename T> void print_matrix(int nrow, int ncol, T *v, int ldv);
template <typename T>
void print_matrix_cuda(int nrow, int ncol, T *dv, int lddv);

template <typename T>
void print_matrix(int nrow, int ncol, int nfib, T *v, int ldv1, int ldv2);
template <typename T>
void print_matrix_cuda(int nrow, int ncol, int nfib, T *dv, int lddv1,
                       int lddv2, int sizex);

template <typename T>
bool compare_matrix(int nrow, int ncol, T *v1, int ldv1, T *v2, int ldv2);
template <typename T>
bool compare_matrix_cuda(int nrow, int ncol, T *dv1, int lddv1, T *dv2,
                         int lddv2);
template <typename T>
bool compare_matrix(int nrow, int ncol, int nfib, T *v1, int ldv11, int ldv12,
                    T *v2, int ldv21, int ldv22, bool print_matrix);
template <typename T>
bool compare_matrix_cuda(int nrow, int ncol, int nfib, T *dv1, int lddv11,
                         int lddv12, int sizex1, T *dv2, int lddv21, int lddv22,
                         int sizex2, bool print_matrix);

template <typename T>
void verify_matrix(int nrow, int ncol, int nfib, T *v, int ldv1, int ldv2,
                   std::string file_prefix, bool store, bool verify);
template <typename T>
void verify_matrix_cuda(int nrow, int ncol, int nfib, T *dv, int lddv1,
                        int lddv2, int sizex, std::string file_prefix,
                        bool store, bool verify);

void cudaMallocHelper(void **devPtr, size_t size);
void cudaMallocPitchHelper(void **devPtr, size_t *pitch, size_t width,
                           size_t height);
void cudaMalloc3DHelper(void **devPtr, size_t *pitch, size_t width,
                        size_t height, size_t depth);
void cudaMallocHostHelper(void **ptr, size_t size);

template <typename T, int D>
void cudaMemcpyAsyncHelper(mgard_cuda_handle<T, D> &handle, void *dst,
                           const void *src, size_t count, enum copy_type kind,
                           int queue_idx);

template <typename T, int D>
void cudaMemcpy2DAsyncHelper(mgard_cuda_handle<T, D> &handle, void *dst,
                             size_t dpitch, void *src, size_t spitch,
                             size_t width, size_t height, enum copy_type kind,
                             int queue_idx);

template <typename T, int D>
void cudaMemcpy3DAsyncHelper(mgard_cuda_handle<T, D> &handle, void *dst,
                             size_t dpitch, size_t dwidth, size_t dheight,
                             void *src, size_t spitch, size_t swidth,
                             size_t sheight, size_t width, size_t height,
                             size_t depth, enum copy_type kind, int queue_idx);

void cudaFreeHelper(void *devPtr);
void cudaFreeHostHelper(void *ptr);
void cudaMemsetHelper(void *devPtr, int value, size_t count);
void cudaMemset2DHelper(void *devPtr, size_t pitch, int value, size_t width,
                        size_t height);
void cudaMemset3DHelper(void *devPtr, size_t pitch, size_t dwidth,
                        size_t dheight, int value, size_t width, size_t height,
                        size_t depth);
void cudaSetDeviceHelper(int dev_id);

template <typename T, int D>
void cudaMemcpyPeerAsyncHelper(mgard_cuda_handle<T, D> &handle, void *dst,
                               int dst_dev, const void *src, int src_dev,
                               size_t count, int queue_idx);

template <typename T, int D>
void cudaMemcpy3DPeerAsyncHelper(mgard_cuda_handle<T, D> &handle, void *dst,
                                 int dst_dev, size_t dpitch, size_t dwidth,
                                 size_t dheight, void *src, int src_dev,
                                 size_t spitch, size_t swidth, size_t sheight,
                                 size_t width, size_t height, size_t depth,
                                 int queue_idx);
bool isGPUPointer(void *ptr);
} // namespace mgard_cuda

#endif