/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */
#include "mgard-x/RuntimeX/RuntimeX.h"
namespace mgard_x {

template void BeginAutoTuning<SYCL>();
template void EndAutoTuning<SYCL>();

AutoTuningTable<SYCL> AutoTuner<SYCL>::autoTuningTable;
bool AutoTuner<SYCL>::ProfileKernels = false;
bool AutoTuner<SYCL>::WriteToTable = false;

} // namespace mgard_x