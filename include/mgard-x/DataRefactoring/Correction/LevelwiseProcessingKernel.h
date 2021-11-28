/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_LEVELWISE_PROCESSING_KERNEL
#define MGARD_X_LEVELWISE_PROCESSING_KERNEL

#include "Common.h"

namespace mgard_x {

template <DIM D, typename T, int OP>
void lwpk(Handle<D, T> &handle, thrust::device_vector<SIZE> shape, T *dv,
          thrust::device_vector<SIZE> ldvs, T *dwork,
          thrust::device_vector<SIZE> ldws, int queue_idx);

template <DIM D, typename T, int OP>
void lwpk(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_d, T *dv, SIZE *ldvs,
          T *dwork, SIZE *ldws, int queue_idx);

} // namespace mgard_x

#endif