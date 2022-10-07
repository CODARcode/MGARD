/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.hpp"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "../Correction/LevelwiseProcessingKernel.hpp"

#ifndef MGARD_X_DATA_REFACTORING_ADD_ND
#define MGARD_X_DATA_REFACTORING_ADD_ND

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void AddND(SubArray<D, T, DeviceType> dinput,
           SubArray<D, T, DeviceType> &doutput, int queue_idx) {

  DeviceLauncher<DeviceType>::Execute(
      LwpkReoKernel<D, T, ADD, DeviceType>(dinput, doutput), queue_idx);
}

} // namespace mgard_x

#endif