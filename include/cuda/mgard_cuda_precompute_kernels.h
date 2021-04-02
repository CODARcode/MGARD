/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/mgard_cuda_common.h"
#include "cuda/mgard_cuda_common_internal.h"

namespace mgard_cuda {
template <typename T, int D>
void calc_cpt_dist(mgard_cuda_handle<T, D> &handle, int n, T *dcoord, T *ddist,
                   int queue_idx);

template <typename T, int D>
void reduce_two_dist(mgard_cuda_handle<T, D> &handle, int n, T *ddist,
                     T *ddist_reduced, int queue_idx);

template <typename T, int D>
void calc_cpt_dist_ratio(mgard_cuda_handle<T, D> &handle, int n, T *dcoord,
                         T *dratio, int queue_idx);

template <typename T, int D>
void dist_to_ratio(mgard_cuda_handle<T, D> &handle, int n, T *ddist, T *dratio,
                   int queue_idx);

template <typename T, int D>
void calc_am_bm(mgard_cuda_handle<T, D> &handle, int n, T *ddist, T *am, T *bm,
                int queue_idx);
} // namespace mgard_cuda