/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "../../../Hierarchy/Hierarchy.h"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "CoefficientKernel.hpp"

#ifndef MGARD_X_160_DATA_REFACTORING_CALC_COEFFICIENTS
#define MGARD_X_160_DATA_REFACTORING_CALC_COEFFICIENTS

namespace mgard_x_160 {

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

} // namespace mgard_x_160

#endif