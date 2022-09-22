/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_LINEAR_QUANTIZATION
#define MGARD_X_LINEAR_QUANTIZATION

#include "Common.h"

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void LinearQuanziation(Hierarchy<D, T, DeviceType>& hierarchy, SubArray<D, T, DeviceType> in_subarray,
                        Config config, enum error_bound_type type, T tol, T s, T norm, LENGTH &outlier_count,
                       CompressionLowLevelWorkspace<D, T, DeviceType> &workspace,
                       int queue_idx);

template <DIM D, typename T>
void levelwise_linear_dequantize(Handle<D, T> &handle, SIZE *shapes,
                                 SIZE l_target, T *volumes, SIZE ldvolumes,
                                 Metadata &m, QUANTIZED_INT *dv, SIZE *ldvs,
                                 T *dwork, SIZE *ldws, bool prep_huffmam,
                                 LENGTH outlier_count, LENGTH *outlier_idx,
                                 QUANTIZED_INT *outliers, int queue_idx);

} // namespace mgard_x

#endif