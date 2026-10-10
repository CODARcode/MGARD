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

#ifndef MGARD_X_160_DATA_REFACTORING_COPY_ND
#define MGARD_X_160_DATA_REFACTORING_COPY_ND

namespace mgard_x_160 {

namespace data_refactoring {

namespace multi_dimension {

template <DIM D, typename T, typename DeviceType>
void Copy3D(SubArray<D, T, DeviceType> dinput,
            SubArray<D, T, DeviceType> doutput, bool padding, int queue_idx) {

  SIZE nf = dinput.shape(D - 1);
  SIZE nc = D >= 2 ? dinput.shape(D - 2) : 1;
  SIZE nr = D >= 3 ? dinput.shape(D - 3) : 1;
  if (padding) {
    DeviceLauncher<DeviceType>::Execute(
        Lwpk3DKernel<D, T, COPY, true, DeviceType>(nf, nc, nr, dinput, doutput),
        queue_idx);
  } else {
    DeviceLauncher<DeviceType>::Execute(
        Lwpk3DKernel<D, T, COPY, false, DeviceType>(nf, nc, nr, dinput,
                                                    doutput),
        queue_idx);
  }
}

} // namespace multi_dimension

} // namespace data_refactoring

} // namespace mgard_x_160

#endif