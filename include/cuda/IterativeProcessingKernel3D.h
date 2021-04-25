/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_ITERATIVE_PROCESSING_KERNEL_3D
#define MGRAD_CUDA_ITERATIVE_PROCESSING_KERNEL_3D

#include "Common.h"
#include "CommonInternal.h"

namespace mgard_cuda {

template <uint32_t D, typename T>
void ipk_1_3d(Handle<D, T> &handle, int nr, int nc, int nf_c, T *am, T *bm,
              T *ddist_f, T *dv, int lddv1, int lddv2, int queue_idx,
              int config);

template <uint32_t D, typename T>
void ipk_2_3d(Handle<D, T> &handle, int nr, int nc_c, int nf_c, T *am, T *bm,
              T *ddist_c, T *dv, int lddv1, int lddv2, int queue_idx,
              int config);

template <uint32_t D, typename T>
void ipk_3_3d(Handle<D, T> &handle, int nr_c, int nc_c, int nf_c, T *am, T *bm,
              T *ddist_r, T *dv, int lddv1, int lddv2, int queue_idx,
              int config);

} // namespace mgard_cuda

#endif