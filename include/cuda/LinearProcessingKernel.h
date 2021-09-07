/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_LINEAR_PROCESSING_KERNEL
#define MGRAD_CUDA_LINEAR_PROCESSING_KERNEL

#include "Common.h"

namespace mgard_cuda {

template <DIM D, typename T>
void lpk_reo_1(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_c_h,
               SIZE *shape_d, SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws,
               DIM processed_n, DIM *processed_dims_h, DIM *processed_dims_d,
               DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f, T *ddist_f,
               T *dratio_f, T *dv1, LENGTH lddv11, LENGTH lddv12, T *dv2,
               LENGTH lddv21, LENGTH lddv22, T *dw, LENGTH lddw1, LENGTH lddw2,
               int queue_idx, int config);

template <DIM D, typename T>
void lpk_reo_2(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_c_h,
               SIZE *shape_d, SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws,
               DIM processed_n, DIM *processed_dims_h, DIM *processed_dims_d,
               DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f, T *ddist_c,
               T *dratio_c, T *dv1, LENGTH lddv11, LENGTH lddv12, T *dv2,
               LENGTH lddv21, LENGTH lddv22, T *dw, LENGTH lddw1, LENGTH lddw2,
               int queue_idx, int config);

template <DIM D, typename T>
void lpk_reo_3(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_c_h,
               SIZE *shape_d, SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws,
               DIM processed_n, DIM *processed_dims_h, DIM *processed_dims_d,
               DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f, T *ddist_r,
               T *dratio_r, T *dv1, LENGTH lddv11, LENGTH lddv12, T *dv2,
               LENGTH lddv21, LENGTH lddv22, T *dw, LENGTH lddw1, LENGTH lddw2,
               int queue_idx, int config);

} // namespace mgard_cuda

#endif