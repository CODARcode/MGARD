/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_ITERATIVE_PROCESSING_KERNEL_3D
#define MGRAD_CUDA_ITERATIVE_PROCESSING_KERNEL_3D

#include "../../Common.h"

namespace mgard_cuda {

template <DIM D, typename T>
void ipk_1_3d(Handle<D, T> &handle, SIZE nr, SIZE nc, SIZE nf_c, T *am, T *bm,
              T *ddist_f, T *dv, SIZE lddv1, SIZE lddv2, int queue_idx,
              int config);

template <DIM D, typename T>
void ipk_2_3d(Handle<D, T> &handle, SIZE nr, SIZE nc_c, SIZE nf_c, T *am, T *bm,
              T *ddist_c, T *dv, SIZE lddv1, SIZE lddv2, int queue_idx,
              int config);

template <DIM D, typename T>
void ipk_3_3d(Handle<D, T> &handle, SIZE nr_c, SIZE nc_c, SIZE nf_c, T *am, T *bm,
              T *ddist_r, T *dv, SIZE lddv1, SIZE lddv2, int queue_idx,
              int config);

} // namespace mgard_cuda

#endif