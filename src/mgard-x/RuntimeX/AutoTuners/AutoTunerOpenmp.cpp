/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */
#include "mgard-x/RuntimeX/RuntimeX.h"
namespace mgard_x {

template void BeginAutoTuning<OPENMP>();
template void EndAutoTuning<OPENMP>();

AutoTuningTable<OPENMP> AutoTuner<OPENMP>::autoTuningTable;
bool AutoTuner<OPENMP>::ProfileKernels = false;
bool AutoTuner<OPENMP>::WriteToTable = false;

} // namespace mgard_x