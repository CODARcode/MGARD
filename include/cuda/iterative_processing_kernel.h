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
void ipk_1(mgard_cuda_handle<T, D> &handle, thrust::device_vector<int> shape,
           thrust::device_vector<int> shape_c, thrust::device_vector<int> ldvs,
           thrust::device_vector<int> ldws,
           thrust::device_vector<int> processed_dims, int curr_dim_r,
           int curr_dim_c, int curr_dim_f, T *am, T *bm, T *ddist_f, T *dv,
           int lddv1, int lddv2, int queue_idx, int config);

template <typename T, int D>
void ipk_2(mgard_cuda_handle<T, D> &handle, thrust::device_vector<int> shape,
           thrust::device_vector<int> shape_c, thrust::device_vector<int> ldvs,
           thrust::device_vector<int> ldws,
           thrust::device_vector<int> processed_dims, int curr_dim_r,
           int curr_dim_c, int curr_dim_f, T *am, T *bm, T *ddist_c, T *dv,
           int lddv1, int lddv2, int queue_idx, int config);

template <typename T, int D>
void ipk_3(mgard_cuda_handle<T, D> &handle, thrust::device_vector<int> shape,
           thrust::device_vector<int> shape_c, thrust::device_vector<int> ldvs,
           thrust::device_vector<int> ldws,
           thrust::device_vector<int> processed_dims, int curr_dim_r,
           int curr_dim_c, int curr_dim_f, T *am, T *bm, T *ddist_r, T *dv,
           int lddv1, int lddv2, int queue_idx, int config);

template <typename T, int D>
void ipk_1(mgard_cuda_handle<T, D> &handle, int *shape_h, int *shape_c_h,
           int *shape_d, int *shape_c_d, int *ldvs, int *ldws, int processed_n,
           int *processed_dims_h, int *processed_dims_d, int curr_dim_r,
           int curr_dim_c, int curr_dim_f, T *am, T *bm, T *ddist_f, T *dv,
           int lddv1, int lddv2, int queue_idx, int config);

template <typename T, int D>
void ipk_2(mgard_cuda_handle<T, D> &handle, int *shape_h, int *shape_c_h,
           int *shape_d, int *shape_c_d, int *ldvs, int *ldws, int processed_n,
           int *processed_dims_h, int *processed_dims_d, int curr_dim_r,
           int curr_dim_c, int curr_dim_f, T *am, T *bm, T *ddist_c, T *dv,
           int lddv1, int lddv2, int queue_idx, int config);

template <typename T, int D>
void ipk_3(mgard_cuda_handle<T, D> &handle, int *shape_h, int *shape_c_h,
           int *shape_d, int *shape_c_d, int *ldvs, int *ldws, int processed_n,
           int *processed_dims_h, int *processed_dims_d, int curr_dim_r,
           int curr_dim_c, int curr_dim_f, T *am, T *bm, T *ddist_r, T *dv,
           int lddv1, int lddv2, int queue_idx, int config);
} // namespace mgard_cuda