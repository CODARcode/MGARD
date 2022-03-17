/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_SINGLE_DIMENSION_DATA_REFACTORING
#define MGARD_X_SINGLE_DIMENSION_DATA_REFACTORING

#include "../../Hierarchy.h"
#include "../../RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void decompose_single(Hierarchy<D, T, DeviceType> &hierarchy,
                      SubArray<D, T, DeviceType> &v, SIZE l_target,
                      int queue_idx);

template <DIM D, typename T, typename DeviceType>
void recompose_single(Hierarchy<D, T, DeviceType> &hierarchy,
                      SubArray<D, T, DeviceType> &v, SIZE l_target,
                      int queue_idx);

} // namespace mgard_x

#endif