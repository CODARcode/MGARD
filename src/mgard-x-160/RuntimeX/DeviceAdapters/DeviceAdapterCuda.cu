/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "mgard-x-160/RuntimeX/RuntimeX.h"

namespace mgard_x_160 {

int cuda_dev_id = 0;
DeviceQueues<CUDA> DeviceRuntime<CUDA>::queues;
DeviceSpecification<CUDA> DeviceRuntime<CUDA>::DeviceSpecs;

bool DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors = false;
bool DeviceRuntime<CUDA>::TimingAllKernels = false;
bool DeviceRuntime<CUDA>::PrintKernelConfig = false;

template <> bool deviceAvailable<CUDA>() {
  return DeviceRuntime<CUDA>::GetDeviceCount() > 0;
}

} // namespace mgard_x_160