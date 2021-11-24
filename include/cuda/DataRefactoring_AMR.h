/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_DATA_REFACTORING_AMR
#define MGARD_X_DATA_REFACTORING_AMR

#include "CommonInternal.h"
#include "MemoryManagement.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace mgard_x {

template <uint32_t D, typename T>
void decompose_amr(Handle<D, T> &handle, T *dv, std::vector<int> ldvs,
               int l_target);

template <uint32_t D, typename T>
void recompose_amr(Handle<D, T> &handle, T *dv, std::vector<int> ldvs,
               int l_target);

} // namespace mgard_x

#endif