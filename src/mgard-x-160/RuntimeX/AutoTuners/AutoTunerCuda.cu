/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */
#include "mgard-x-160/RuntimeX/RuntimeX.h"
namespace mgard_x_160 {

template void BeginAutoTuning<CUDA>();
template void EndAutoTuning<CUDA>();

AutoTuningTable<CUDA> AutoTuner<CUDA>::autoTuningTable;
bool AutoTuner<CUDA>::ProfileKernels = false;
bool AutoTuner<CUDA>::WriteToTable = false;

} // namespace mgard_x_160