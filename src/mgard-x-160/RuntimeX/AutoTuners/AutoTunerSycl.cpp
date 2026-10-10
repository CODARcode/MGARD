/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */
#include "mgard-x-160/RuntimeX/RuntimeX.h"
namespace mgard_x_160 {

template void BeginAutoTuning<SYCL>();
template void EndAutoTuning<SYCL>();

AutoTuningTable<SYCL> AutoTuner<SYCL>::autoTuningTable;
bool AutoTuner<SYCL>::ProfileKernels = false;
bool AutoTuner<SYCL>::WriteToTable = false;

} // namespace mgard_x_160