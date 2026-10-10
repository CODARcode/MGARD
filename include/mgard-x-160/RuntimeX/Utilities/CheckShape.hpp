/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_CHECK_SHAPE_HPP
#define MGARD_X_160_CHECK_SHAPE_HPP

namespace mgard_x_160 {

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

template <typename T> T roundup(T a, T b) {
  return ((double)(a - 1) / b + 1) * b;
}

} // namespace mgard_x_160

#endif