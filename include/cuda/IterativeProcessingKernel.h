/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_ITERATIVE_PROCESSING_KERNEL
#define MGRAD_CUDA_ITERATIVE_PROCESSING_KERNEL

#include "Common.h"

namespace mgard_cuda {

template <uint32_t D, typename T>
void ipk_1(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_c_h, SIZE *shape_d,
           SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM processed_n,
           DIM *processed_dims_h, DIM *processed_dims_d, DIM curr_dim_r,
           DIM curr_dim_c, DIM curr_dim_f, T *am, T *bm, T *ddist_f, T *dv,
           LENGTH lddv1, LENGTH lddv2, int queue_idx, int config);

template <uint32_t D, typename T>
void ipk_2(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_c_h, SIZE *shape_d,
           SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM processed_n,
           DIM *processed_dims_h, DIM *processed_dims_d, DIM curr_dim_r,
           DIM curr_dim_c, DIM curr_dim_f, T *am, T *bm, T *ddist_c, T *dv,
           LENGTH lddv1, LENGTH lddv2, int queue_idx, int config);

template <uint32_t D, typename T>
void ipk_3(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_c_h, SIZE *shape_d,
           SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM processed_n,
           DIM *processed_dims_h, DIM *processed_dims_d, DIM curr_dim_r,
           DIM curr_dim_c, DIM curr_dim_f, T *am, T *bm, T *ddist_r, T *dv,
           LENGTH lddv1, LENGTH lddv2, int queue_idx, int config);
} // namespace mgard_cuda

#endif