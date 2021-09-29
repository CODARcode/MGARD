/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGARD_CUDA_MEMORY_MANAGEMENT
#define MGARD_CUDA_MEMORY_MANAGEMENT

#include "Handle.h"

#include <string>

namespace mgard_cuda {

enum copy_type { H2D, D2H, D2D, H2H, AUTO };

enum endiness_type CheckEndianess();

template <typename SubArrayType>
void PrintSubarray(std::string name, SubArrayType subArray);

template <typename T> void print_matrix(SIZE nrow, SIZE ncol, T *v, SIZE ldv);
template <typename T>
void print_matrix_cuda(SIZE nrow, SIZE ncol, T *dv, SIZE lddv);

template <typename T>
void print_matrix(SIZE nrow, SIZE ncol, SIZE nfib, T *v, SIZE ldv1, SIZE ldv2);
template <typename T>
void print_matrix_cuda(SIZE nrow, SIZE ncol, SIZE nfib, T *dv, SIZE lddv1,
                       SIZE lddv2, SIZE sizex);

template <typename T>
bool compare_matrix(SIZE nrow, SIZE ncol, T *v1, SIZE ldv1, T *v2, SIZE ldv2);
template <typename T>
bool compare_matrix_cuda(SIZE nrow, SIZE ncol, T *dv1, SIZE lddv1, T *dv2,
                         SIZE lddv2);
template <typename T>
bool compare_matrix(SIZE nrow, SIZE ncol, SIZE nfib, T *v1, SIZE ldv11,
                    SIZE ldv12, T *v2, SIZE ldv21, SIZE ldv22,
                    bool print_matrix);
template <typename T>
bool compare_matrix_cuda(SIZE nrow, SIZE ncol, SIZE nfib, T *dv1, SIZE lddv11,
                         SIZE lddv12, SIZE sizex1, T *dv2, SIZE lddv21,
                         SIZE lddv22, SIZE sizex2, bool print_matrix);

template <typename T>
void verify_matrix(SIZE nrow, SIZE ncol, SIZE nfib, T *v, SIZE ldv1, SIZE ldv2,
                   std::string file_prefix, bool store, bool verify);
template <typename T>
void verify_matrix_cuda(SIZE nrow, SIZE ncol, SIZE nfib, T *dv, SIZE lddv1,
                        SIZE lddv2, SIZE sizex, std::string file_prefix,
                        bool store, bool verify);

template <DIM D, typename T>
void cudaMallocHelper(Handle<D, T> &handle, void **devPtr, size_t size_t);

template <DIM D, typename T>
void cudaMallocPitchHelper(Handle<D, T> &handle, void **devPtr, size_t *pitch,
                           size_t width, size_t height);

template <DIM D, typename T>
void cudaMalloc3DHelper(Handle<D, T> &handle, void **devPtr, size_t *pitch,
                        size_t width, size_t height, size_t depth);
void cudaMallocHostHelper(void **ptr, size_t size_t);

template <DIM D, typename T>
void cudaMemcpyAsyncHelper(Handle<D, T> &handle, void *dst, const void *src,
                           size_t count, enum copy_type kind, int queue_idx);

template <DIM D, typename T>
void cudaMemcpy2DAsyncHelper(Handle<D, T> &handle, void *dst, size_t dpitch,
                             void *src, size_t spitch, size_t width,
                             size_t height, enum copy_type kind, int queue_idx);

template <DIM D, typename T>
void cudaMemcpy3DAsyncHelper(Handle<D, T> &handle, void *dst, size_t dpitch,
                             size_t dwidth, size_t dheight, const void *src,
                             size_t spitch, size_t swidth, size_t sheight,
                             size_t width, size_t height, size_t depth,
                             enum copy_type kind, int queue_idx);

void cudaFreeHelper(void *devPtr);
void cudaFreeHostHelper(void *ptr);
void cudaMemsetHelper(void *devPtr, int value, size_t count);
void cudaMemset2DHelper(void *devPtr, size_t pitch, int value, size_t width,
                        size_t height);
void cudaMemset3DHelper(void *devPtr, size_t pitch, size_t dwidth,
                        size_t dheight, int value, size_t width, size_t height,
                        size_t depth);
void cudaSetDeviceHelper(int dev_id);

template <DIM D, typename T>
void cudaMemcpyPeerAsyncHelper(Handle<D, T> &handle, void *dst, int dst_dev,
                               const void *src, int src_dev, size_t count,
                               int queue_idx);

template <DIM D, typename T>
void cudaMemcpy3DPeerAsyncHelper(Handle<D, T> &handle, void *dst, int dst_dev,
                                 size_t dpitch, size_t dwidth, size_t dheight,
                                 const void *src, int src_dev, size_t spitch,
                                 size_t swidth, size_t sheight, size_t width,
                                 size_t height, size_t depth, int queue_idx);
bool isGPUPointer(void *ptr);
} // namespace mgard_cuda

#endif