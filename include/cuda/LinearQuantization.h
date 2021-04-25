/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_LINEAR_QUANTIZATION
#define MGRAD_CUDA_LINEAR_QUANTIZATION

#include "Common.h"
#include "CommonInternal.h"

namespace mgard_cuda {

template <uint32_t D, typename T>
void levelwise_linear_quantize(Handle<D, T> &handle, int *shapes, int l_target,
                               quant_meta<T> m, T *dv, int *ldvs, int *dwork,
                               int *ldws, bool prep_huffmam, int *shape,
                               size_t *outlier_count, unsigned int *outlier_idx,
                               int *outliers, int queue_idx);

template <uint32_t D, typename T>
void levelwise_linear_dequantize(Handle<D, T> &handle, int *shapes,
                                 int l_target, quant_meta<T> m, int *dv,
                                 int *ldvs, T *dwork, int *ldws,
                                 size_t outlier_count,
                                 unsigned int *outlier_idx, int *outliers,
                                 int queue_idx);

} // namespace mgard_cuda

#endif