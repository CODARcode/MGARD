/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "mgard-x/DataRefactoring/MultiDimension/DataRefactoring.hpp"
#include "mgard-x/DataRefactoring/SingleDimension/DataRefactoring.hpp"

#include <iostream>

#include <chrono>
namespace mgard_x {

template void recompose<1, double, HIP>(Hierarchy<1, double, HIP> &hierarchy,
                                        SubArray<1, double, HIP> &v,
                                        SIZE l_target, int queue_idx);
template void
    recompose_single<1, double, HIP>(Hierarchy<1, double, HIP> &hierarchy,
                                     SubArray<1, double, HIP> &v, SIZE l_target,
                                     int queue_idx);
} // namespace mgard_x