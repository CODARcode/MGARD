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
class SingleDimensionCoefficientFunctor: public Functor<DeviceType> {
public:
  MGARDX_CONT SingleDimensionCoefficientFunctor() {}
  MGARDX_CONT SingleDimensionCoefficientFunctor(SubArray<1, SIZE, DeviceType> shape, 
                                                SubArray<1, SIZE, DeviceType> shape_c,
                                                SIZE current_dim,
                                                SubArray<1, T, DeviceType> ratio,
                                                SubArray<D, T, DeviceType> v):
                              shape(shape), shape_c(shape_c), current_dim(current_dim),
                              ratio(ratio), v(v){
                                Functor<DeviceType>();
                              }

  MGARDX_EXEC void
  Operation1() {
    
  }


private:
  // functor parameters
  SubArray<1, SIZE, DeviceType> shape;
  SubArray<1, SIZE, DeviceType> shape_c;
  SIZE current_dim;
  SubArray<1, T, DeviceType> ratio;
  SubArray<D, T, DeviceType> v;

}


}

#endif