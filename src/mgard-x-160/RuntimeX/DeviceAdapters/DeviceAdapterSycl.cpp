/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "mgard-x-160/RuntimeX/RuntimeX.h"

namespace mgard_x_160 {

int sycl_dev_id = 0;
DeviceQueues<SYCL> DeviceRuntime<SYCL>::queues;
DeviceSpecification<SYCL> DeviceRuntime<SYCL>::DeviceSpecs;

// SyncAllKernelsAndCheckErrors needs to be always ON for SYCL
bool DeviceRuntime<SYCL>::SyncAllKernelsAndCheckErrors = true;
bool DeviceRuntime<SYCL>::TimingAllKernels = false;
bool DeviceRuntime<SYCL>::PrintKernelConfig = false;

template <> bool deviceAvailable<SYCL>() {
  return DeviceRuntime<SYCL>::GetDeviceCount() > 0;
}

} // namespace mgard_x_160