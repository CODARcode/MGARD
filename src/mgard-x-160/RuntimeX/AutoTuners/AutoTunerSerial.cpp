/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */
#include "mgard-x-160/RuntimeX/RuntimeX.h"
namespace mgard_x_160 {

template void BeginAutoTuning<SERIAL>();
template void EndAutoTuning<SERIAL>();

AutoTuningTable<SERIAL> AutoTuner<SERIAL>::autoTuningTable;
bool AutoTuner<SERIAL>::ProfileKernels = false;
bool AutoTuner<SERIAL>::WriteToTable = false;

} // namespace mgard_x_160