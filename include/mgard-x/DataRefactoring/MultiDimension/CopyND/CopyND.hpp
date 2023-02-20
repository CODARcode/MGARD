/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.h"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "../Correction/LevelwiseProcessingKernel.hpp"

#ifndef MGARD_X_DATA_REFACTORING_COPY_ND
#define MGARD_X_DATA_REFACTORING_COPY_ND

namespace mgard_x {

namespace data_refactoring {

namespace multi_dimension {

template <DIM D, typename T, typename DeviceType>
void CopyND(SubArray<D, T, DeviceType> dinput,
            SubArray<D, T, DeviceType> doutput, int queue_idx) {

  DeviceLauncher<DeviceType>::Execute(
      LwpkReoKernel<D, T, COPY, DeviceType>(dinput, doutput), queue_idx);
}

} // namespace multi_dimension

} // namespace data_refactoring

} // namespace mgard_x

#endif