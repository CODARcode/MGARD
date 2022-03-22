/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

int DeviceRuntime<SYCL>::curr_dev_id = 0;
DeviceQueues<SYCL> DeviceRuntime<SYCL>::queues;
DeviceSpecification<SYCL> DeviceRuntime<SYCL>::DeviceSpecs;

bool DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors = false;
bool MemoryManager<SYCL>::ReduceMemoryFootprint = false;
bool DeviceRuntime<SYCL>::TimingAllKernels = false;
bool DeviceRuntime<SYCL>::PrintKernelConfig = false;

AutoTuningTable<SYCL> AutoTuner<SYCL>::autoTuningTable;
bool AutoTuner<SYCL>::ProfileKernels = false;

template <> bool deviceAvailable<SYCL>() {
  return DeviceRuntime<SYCL>::GetDeviceCount() > 0;
}

} // namespace mgard_x