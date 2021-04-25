/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGARD_CUDA_PRECOMPUTE_KERNELS
#define MGARD_CUDA_PRECOMPUTE_KERNELS

#include "Common.h"
#include "CommonInternal.h"

namespace mgard_cuda {
template <uint32_t D, typename T>
void calc_cpt_dist(Handle<D, T> &handle, int n, T *dcoord, T *ddist,
                   int queue_idx);

template <uint32_t D, typename T>
void reduce_two_dist(Handle<D, T> &handle, int n, T *ddist, T *ddist_reduced,
                     int queue_idx);

template <uint32_t D, typename T>
void calc_cpt_dist_ratio(Handle<D, T> &handle, int n, T *dcoord, T *dratio,
                         int queue_idx);

template <uint32_t D, typename T>
void dist_to_ratio(Handle<D, T> &handle, int n, T *ddist, T *dratio,
                   int queue_idx);

template <uint32_t D, typename T>
void calc_am_bm(Handle<D, T> &handle, int n, T *ddist, T *am, T *bm,
                int queue_idx);
} // namespace mgard_cuda

#endif