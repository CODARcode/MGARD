/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_PRECOMPUTE_KERNELS
#define MGARD_X_PRECOMPUTE_KERNELS

#include "Common.h"

namespace mgard_x {
template <DIM D, typename T>
void calc_cpt_dist(Handle<D, T> &handle, int n, T *dcoord, T *ddist,
                   int queue_idx);

template <DIM D, typename T>
void reduce_two_dist(Handle<D, T> &handle, int n, T *ddist, T *ddist_reduced,
                     int queue_idx);

template <DIM D, typename T>
void dist_to_ratio(Handle<D, T> &handle, int n, T *ddist, T *dratio,
                   int queue_idx);

template <DIM D, typename T>
void dist_to_volume(Handle<D, T> &handle, int n, T *ddist, T *dvolume,
                   int queue_idx);

template <DIM D, typename T>
void calc_am_bm(Handle<D, T> &handle, int n, T *ddist, T *am, T *bm,
                int queue_idx);
} // namespace mgard_x

#endif