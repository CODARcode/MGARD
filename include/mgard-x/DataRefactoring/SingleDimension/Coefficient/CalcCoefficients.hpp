/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.h"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "CoefficientKernel.hpp"

#ifndef MGARD_X_DATA_REFACTORING_CALC_COEFFICIENTS
#define MGARD_X_DATA_REFACTORING_CALC_COEFFICIENTS

namespace mgard_x {

namespace data_refactoring {

namespace single_dimension {

template <DIM D, typename T, typename DeviceType>
void CalcCoefficients(DIM current_dim, SubArray<1, T, DeviceType> ratio,
                      SubArray<D, T, DeviceType> v,
                      SubArray<D, T, DeviceType> coarse,
                      SubArray<D, T, DeviceType> coeff, int queue_idx) {

  DeviceLauncher<DeviceType>::Execute(
      SingleDimensionCoefficientKernel<D, T, DECOMPOSE, DeviceType>(
          current_dim, ratio, v, coarse, coeff),
      queue_idx);
}

} // namespace single_dimension

} // namespace data_refactoring

} // namespace mgard_x

#endif