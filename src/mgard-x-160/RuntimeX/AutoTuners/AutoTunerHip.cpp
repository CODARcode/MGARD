/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */
#include "mgard-x-160/RuntimeX/RuntimeX.h"
namespace mgard_x_160 {

template void BeginAutoTuning<HIP>();
template void EndAutoTuning<HIP>();

AutoTuningTable<HIP> AutoTuner<HIP>::autoTuningTable;
bool AutoTuner<HIP>::ProfileKernels = false;
bool AutoTuner<HIP>::WriteToTable = false;

} // namespace mgard_x_160