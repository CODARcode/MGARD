/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_LINEAR_PROCESSING_KERNEL_3D
#define MGRAD_CUDA_LINEAR_PROCESSING_KERNEL_3D

#include "Common.h"
#include "CommonInternal.h"

namespace mgard_cuda {

template <uint32_t D, typename T>
void lpk_reo_1_3d(Handle<D, T> &handle, int nr, int nc, int nf, int nf_c,
                  int zero_r, int zero_c, int zero_f, T *ddist_f, T *dratio_f,
                  T *dv1, int lddv11, int lddv12, T *dv2, int lddv21,
                  int lddv22, T *dw, int lddw1, int lddw2, int queue_idx,
                  int config);

template <uint32_t D, typename T>
void lpk_reo_2_3d(Handle<D, T> &handle, int nr, int nc, int nf_c, int nc_c,
                  T *ddist_c, T *dratio_c, T *dv1, int lddv11, int lddv12,
                  T *dv2, int lddv21, int lddv22, T *dw, int lddw1, int lddw2,
                  int queue_idx, int config);

template <uint32_t D, typename T>
void lpk_reo_3_3d(Handle<D, T> &handle, int nr, int nc_c, int nf_c, int nr_c,
                  T *ddist_r, T *dratio_r, T *dv1, int lddv11, int lddv12,
                  T *dv2, int lddv21, int lddv22, T *dw, int lddw1, int lddw2,
                  int queue_idx, int config);

} // namespace mgard_cuda

#endif