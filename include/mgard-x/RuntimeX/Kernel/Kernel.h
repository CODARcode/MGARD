/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */
#ifndef MGARD_X_KERNEL
#define MGARD_X_KERNEL

namespace mgard_x {
class Kernel {
public:
  constexpr static bool EnableAutoTuning() { return true; }
};
} // namespace mgard_x
#endif