/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_LINEAR_PROCESSING_KERNEL_3D
#define MGARD_X_LINEAR_PROCESSING_KERNEL_3D

#include "../../Common.h"

namespace mgard_x {

template <DIM D, typename T>
void lpk_reo_1_3d(Handle<D, T> &handle, SIZE nr, SIZE nc, SIZE nf, SIZE nf_c,
                  SIZE zero_r, SIZE zero_c, SIZE zero_f, T *ddist_f, T *dratio_f,
                  T *dv1, SIZE lddv11, SIZE lddv12, T *dv2, SIZE lddv21,
                  SIZE lddv22, T *dw, SIZE lddw1, SIZE lddw2, int queue_idx,
                  int config);

template <DIM D, typename T>
void lpk_reo_2_3d(Handle<D, T> &handle, SIZE nr, SIZE nc, SIZE nf_c, SIZE nc_c,
                  T *ddist_c, T *dratio_c, T *dv1, SIZE lddv11, SIZE lddv12,
                  T *dv2, SIZE lddv21, SIZE lddv22, T *dw, SIZE lddw1, SIZE lddw2,
                  int queue_idx, int config);

template <DIM D, typename T>
void lpk_reo_3_3d(Handle<D, T> &handle, SIZE nr, SIZE nc_c, SIZE nf_c, SIZE nr_c,
                  T *ddist_r, T *dratio_r, T *dv1, SIZE lddv11, SIZE lddv12,
                  T *dv2, SIZE lddv21, SIZE lddv22, T *dw, SIZE lddw1, SIZE lddw2,
                  int queue_idx, int config);

} // namespace mgard_x

#endif