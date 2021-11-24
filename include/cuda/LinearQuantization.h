/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_LINEAR_QUANTIZATION
#define MGARD_X_LINEAR_QUANTIZATION

#include "Common.h"

namespace mgard_x {

template <DIM D, typename T>
void levelwise_linear_quantize(Handle<D, T> &handle, SIZE *shapes,
                               SIZE l_target, T *volumes, SIZE ldvolumes,
                               Metadata &m, T *dv, SIZE *ldvs,
                               QUANTIZED_INT *dwork, SIZE *ldws,
                               bool prep_huffmam, SIZE *shape,
                               LENGTH *outlier_count, LENGTH *outlier_idx,
                               QUANTIZED_INT *outliers, int queue_idx);

template <DIM D, typename T>
void levelwise_linear_dequantize(Handle<D, T> &handle, SIZE *shapes,
                                 SIZE l_target, T *volumes, SIZE ldvolumes,
                                 Metadata &m, QUANTIZED_INT *dv, SIZE *ldvs,
                                 T *dwork, SIZE *ldws, bool prep_huffmam,
                                 LENGTH outlier_count, LENGTH *outlier_idx,
                                 QUANTIZED_INT *outliers, int queue_idx);

} // namespace mgard_x

#endif