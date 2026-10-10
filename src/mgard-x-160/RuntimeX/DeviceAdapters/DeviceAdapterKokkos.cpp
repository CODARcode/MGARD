/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "mgard-x-160/RuntimeX/RuntimeX.h"

namespace mgard_x_160 {

int DeviceRuntime<KOKKOS>::curr_dev_id = 0;
// DeviceQueues<CUDA> DeviceRuntime<CUDA>::queues;
DeviceSpecification<KOKKOS> DeviceRuntime<KOKKOS>::DeviceSpecs;

bool DeviceRuntime<KOKKOS>::SyncAllKernelsAndCheckErrors = false;
// bool MemoryManager<CUDA>::ReduceMemoryFootprint = false;

// KernelConfigs<CUDA> AutoTuner<CUDA>::kernelConfigs;
// AutoTuningTable<CUDA> AutoTuner<CUDA>::autoTuningTable;
// bool AutoTuner<CUDA>::ProfileKernels = false;

} // namespace mgard_x_160