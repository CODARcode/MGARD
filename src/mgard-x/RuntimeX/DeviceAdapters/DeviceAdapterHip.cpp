/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

int DeviceRuntime<HIP>::curr_dev_id = 0;
DeviceQueues<HIP> DeviceRuntime<HIP>::queues;
DeviceSpecification<HIP> DeviceRuntime<HIP>::DeviceSpecs;

bool DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors = false;
bool MemoryManager<HIP>::ReduceMemoryFootprint = false;
bool DeviceRuntime<HIP>::TimingAllKernels = false;
bool DeviceRuntime<HIP>::PrintKernelConfig = false;

KernelConfigs<HIP> AutoTuner<HIP>::kernelConfigs;
AutoTuningTable<HIP> AutoTuner<HIP>::autoTuningTable;
bool AutoTuner<HIP>::ProfileKernels = false;

template <> bool deviceAvailable<HIP>() {
  return DeviceRuntime<HIP>::GetDeviceCount() > 0;
}

} // namespace mgard_x