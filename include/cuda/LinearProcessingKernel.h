/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_LINEAR_PROCESSING_KERNEL
#define MGRAD_CUDA_LINEAR_PROCESSING_KERNEL

#include "Common.h"
#include "CommonInternal.h"

namespace mgard_cuda {

template <uint32_t D, typename T>
void lpk_reo_1(Handle<D, T> &handle, thrust::device_vector<int> shape,
               thrust::device_vector<int> shape_c,
               thrust::device_vector<int> ldvs, thrust::device_vector<int> ldws,
               thrust::device_vector<int> processed_dims, int curr_dim_r,
               int curr_dim_c, int curr_dim_f, T *ddist_f, T *dratio_f, T *dv1,
               int lddv11, int lddv12, T *dv2, int lddv21, int lddv22, T *dw,
               int lddw1, int lddw2, int queue_idx, int config);

template <uint32_t D, typename T>
void lpk_reo_2(Handle<D, T> &handle, thrust::device_vector<int> shape,
               thrust::device_vector<int> shape_c,
               thrust::device_vector<int> ldvs, thrust::device_vector<int> ldws,
               thrust::device_vector<int> processed_dims, int curr_dim_r,
               int curr_dim_c, int curr_dim_f, T *ddist_c, T *dratio_c, T *dv1,
               int lddv11, int lddv12, T *dv2, int lddv21, int lddv22, T *dw,
               int lddw1, int lddw2, int queue_idx, int config);

template <uint32_t D, typename T>
void lpk_reo_3(Handle<D, T> &handle, thrust::device_vector<int> shape,
               thrust::device_vector<int> shape_c,
               thrust::device_vector<int> ldvs, thrust::device_vector<int> ldws,
               thrust::device_vector<int> processed_dims, int curr_dim_r,
               int curr_dim_c, int curr_dim_f, T *ddist_r, T *dratio_r, T *dv1,
               int lddv11, int lddv12, T *dv2, int lddv21, int lddv22, T *dw,
               int lddw1, int lddw2, int queue_idx, int config);

template <uint32_t D, typename T>
void lpk_reo_1(Handle<D, T> &handle, int *shape_h, int *shape_c_h, int *shape_d,
               int *shape_c_d, int *ldvs, int *ldws, int processed_n,
               int *processed_dims_h, int *processed_dims_d, int curr_dim_r,
               int curr_dim_c, int curr_dim_f, T *ddist_f, T *dratio_f, T *dv1,
               int lddv11, int lddv12, T *dv2, int lddv21, int lddv22, T *dw,
               int lddw1, int lddw2, int queue_idx, int config);

template <uint32_t D, typename T>
void lpk_reo_2(Handle<D, T> &handle, int *shape_h, int *shape_c_h, int *shape_d,
               int *shape_c_d, int *ldvs, int *ldws, int processed_n,
               int *processed_dims_h, int *processed_dims_d, int curr_dim_r,
               int curr_dim_c, int curr_dim_f, T *ddist_c, T *dratio_c, T *dv1,
               int lddv11, int lddv12, T *dv2, int lddv21, int lddv22, T *dw,
               int lddw1, int lddw2, int queue_idx, int config);

template <uint32_t D, typename T>
void lpk_reo_3(Handle<D, T> &handle, int *shape_h, int *shape_c_h, int *shape_d,
               int *shape_c_d, int *ldvs, int *ldws, int processed_n,
               int *processed_dims_h, int *processed_dims_d, int curr_dim_r,
               int curr_dim_c, int curr_dim_f, T *ddist_r, T *dratio_r, T *dv1,
               int lddv11, int lddv12, T *dv2, int lddv21, int lddv22, T *dw,
               int lddw1, int lddw2, int queue_idx, int config);

} // namespace mgard_cuda

#endif