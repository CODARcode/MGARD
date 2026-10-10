/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "../../../Hierarchy/Hierarchy.h"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "LevelwiseProcessingKernel.hpp"

#ifndef MGARD_X_160_DATA_REFACTORING_ADD_ND
#define MGARD_X_160_DATA_REFACTORING_ADD_ND

namespace mgard_x_160 {

namespace data_refactoring {

namespace multi_dimension {

template <DIM D, typename T, typename DeviceType>
void AddND(SubArray<D, T, DeviceType> dinput,
           SubArray<D, T, DeviceType> doutput, int queue_idx) {

  DeviceLauncher<DeviceType>::Execute(
      LwpkReoKernel<D, T, ADD, DeviceType>(dinput, doutput), queue_idx);
}

} // namespace multi_dimension

} // namespace data_refactoring

} // namespace mgard_x_160

#endif