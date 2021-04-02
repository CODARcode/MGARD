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
void ipk_1_3d(mgard_cuda_handle<T, D> &handle, int nr, int nc, int nf_c, T *am,
              T *bm, T *ddist_f, T *dv, int lddv1, int lddv2, int queue_idx,
              int config);

template <typename T, int D>
void ipk_2_3d(mgard_cuda_handle<T, D> &handle, int nr, int nc_c, int nf_c,
              T *am, T *bm, T *ddist_c, T *dv, int lddv1, int lddv2,
              int queue_idx, int config);

template <typename T, int D>
void ipk_3_3d(mgard_cuda_handle<T, D> &handle, int nr_c, int nc_c, int nf_c,
              T *am, T *bm, T *ddist_r, T *dv, int lddv1, int lddv2,
              int queue_idx, int config);

} // namespace mgard_cuda