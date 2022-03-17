/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_CHECK_SHAPE_HPP
#define MGARD_X_CHECK_SHAPE_HPP

namespace mgard_x {

template <DIM D> int check_shape(std::vector<SIZE> shape) {
  if (D != shape.size()) {
    return -1;
  }
  for (DIM i = 0; i < shape.size(); i++) {
    if (shape[i] < 3)
      return -2;
  }
  return 0;
}

} // namespace mgard_x

#endif