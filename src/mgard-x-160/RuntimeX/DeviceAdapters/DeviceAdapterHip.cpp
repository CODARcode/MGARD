/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "mgard-x-160/RuntimeX/RuntimeX.h"

namespace mgard_x_160 {

int hip_dev_id = 0;
DeviceQueues<HIP> DeviceRuntime<HIP>::queues;
DeviceSpecification<HIP> DeviceRuntime<HIP>::DeviceSpecs;

bool DeviceRuntime<HIP>::SyncAllKernelsAndCheckErrors = false;
bool DeviceRuntime<HIP>::TimingAllKernels = false;
bool DeviceRuntime<HIP>::PrintKernelConfig = false;

template <> bool deviceAvailable<HIP>() {
  return DeviceRuntime<HIP>::GetDeviceCount() > 0;
}

} // namespace mgard_x_160