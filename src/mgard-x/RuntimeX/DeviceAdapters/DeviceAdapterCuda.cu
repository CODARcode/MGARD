/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

DeviceQueues<CUDA> DeviceRuntime<CUDA>::queues;
DeviceSpecification<CUDA> DeviceRuntime<CUDA>::DeviceSpecs;

bool DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors = false;
bool MemoryManager<CUDA>::ReduceMemoryFootprint = true;
bool DeviceRuntime<CUDA>::TimingAllKernels = false;
bool DeviceRuntime<CUDA>::PrintKernelConfig = false;

AutoTuningTable<CUDA> AutoTuner<CUDA>::autoTuningTable;
bool AutoTuner<CUDA>::ProfileKernels = false;

template <> bool deviceAvailable<CUDA>() {
  return DeviceRuntime<CUDA>::GetDeviceCount() > 0;
}

} // namespace mgard_x