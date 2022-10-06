/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */
#include "mgard-x/RuntimeX/RuntimeX.h"

#include <string_view>

namespace mgard_x {

template <typename T> MGARDX_CONT constexpr int FunctorNameToAutoTuningTable(std::string functor_name) {
  if constexpr (std::is_same<T, float>::value) {
    return 0;
  } else if constexpr (std::is_same<T, double>::value) {
    return 1;
  } else {
    return 0;
  }
}

} // namespace mgard_x