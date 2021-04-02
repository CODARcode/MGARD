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
void levelwise_linear_quantize(mgard_cuda_handle<T, D> &handle, int *shapes,
                               int l_target, quant_meta<T> m, T *dv, int *ldvs,
                               int *dwork, int *ldws, bool prep_huffmam,
                               int *shape, size_t *outlier_count,
                               unsigned int *outlier_idx, int *outliers,
                               int queue_idx);

template <typename T, int D>
void levelwise_linear_dequantize(mgard_cuda_handle<T, D> &handle, int *shapes,
                                 int l_target, quant_meta<T> m, int *dv,
                                 int *ldvs, T *dwork, int *ldws,
                                 size_t outlier_count,
                                 unsigned int *outlier_idx, int *outliers,
                                 int queue_idx);

} // namespace mgard_cuda