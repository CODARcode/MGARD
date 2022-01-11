/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */


#define MGARDX_COMPILE_HIP
#include "mgard-x/DataRefactoring/MultiDimension/DataRefactoring.hpp"

#include <iostream>

#include <chrono>
namespace mgard_x {

template void decompose<5, float, HIP>(Hierarchy<5, float, HIP> & hierarchy,
                                      SubArray<5, float, HIP>& v,
                                      SIZE l_target, int queue_idx);
} // namespace mgard_x
#undef MGARDX_COMPILE_HIP


