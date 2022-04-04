/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../../../Hierarchy/Hierarchy.hpp"
#include "../../../RuntimeX/RuntimeX.h"

#include "../DataRefactoring.h"

#include "CoefficientKernel.hpp"

#ifndef MGARD_X_DATA_REFACTORING_COEFFICIENTS_RESTORE
#define MGARD_X_DATA_REFACTORING_COEFFICIENTS_RESTORE

namespace mgard_x {

template <DIM D, typename T, typename DeviceType>
void CoefficientsRestore(DIM current_dim, SubArray<1, T, DeviceType> ratio,
                         SubArray<D, T, DeviceType> v,
                         SubArray<D, T, DeviceType> coarse,
                         SubArray<D, T, DeviceType> coeff, int queue_idx) {

  SingleDimensionCoefficient<D, T, RECOMPOSE, DeviceType>().Execute(
      current_dim, ratio, v, coarse, coeff, queue_idx);
}

} // namespace mgard_x

#endif