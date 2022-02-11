/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_SINGLE_DIMENSION_COEFFICIENT_KERNEL_TEMPLATE
#define MGARD_X_SINGLE_DIMENSION_COEFFICIENT_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

#include "../MultiDimension/Coefficient/GPKFunctor.h"
// #include "GridProcessingKernel.h"

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class SingleDimensionCoefficientFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SingleDimensionCoefficientFunctor() {}
  MGARDX_CONT SingleDimensionCoefficientFunctor(
      SubArray<1, SIZE, DeviceType> coeff_shape, DIM current_dim,
      SubArray<1, T, DeviceType> ratio, SubArray<D, T, DeviceType> v,

      SubArray<D, T, DeviceType> coeff)
      : coeff_shape(coeff_shape), shape_c(shape_c), current_dim(current_dim),
        ratio(ratio), v(v), coeff(coeff) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE coeff_idx[D];
    SIZE firstD = div_roundup(*coeff_shape(0), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    coeff_idx[0] =
        (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    bidx /= firstD;
    if (D >= 2)
      coeff_idx[1] = FunctorBase<DeviceType>::GetBlockIdY() *
                         FunctorBase<DeviceType>::GetBlockDimY() +
                     FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      coeff_idx[2] = FunctorBase<DeviceType>::GetBlockIdZ() *
                         FunctorBase<DeviceType>::GetBlockDimZ() +
                     FunctorBase<DeviceType>::GetThreadIdZ();

    for (DIM d = 3; d < D; d++) {
      coeff_idx[d] = bidx % *coeff_shape(d);
      bidx /= shape_sm[d];
    }

    bool in_range = true;
    for (DIM d = 0; d < D; d++) {
      if (coeff_idx[d] >= *coeff_shape(d))
        in_range = false;
    }

    if (in_range) {
      SIZE v_idx[D];
      for (DIM d = 0; d < D; d++) {
        if (d == current_dim)
          v_idx[d] = coeff_idx[d];
      }
      *coeff(coeff_idx) =
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  // functor parameters
  SubArray<1, SIZE, DeviceType> coeff_shape;
  DIM current_dim;
  SubArray<1, T, DeviceType> ratio;
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coeff;
}

} // namespace mgard_x

#endif