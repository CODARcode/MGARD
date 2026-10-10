/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */
#ifndef MGARD_X_160_KERNEL
#define MGARD_X_160_KERNEL

namespace mgard_x_160 {
class Kernel {
public:
  constexpr static bool EnableConfig() { return true; }
  constexpr static bool EnableAutoTuning() { return true; }
};
} // namespace mgard_x_160
#endif