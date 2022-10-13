/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

int serial_dev_id = 0;
DeviceQueues<SERIAL> DeviceRuntime<SERIAL>::queues;
DeviceSpecification<SERIAL> DeviceRuntime<SERIAL>::DeviceSpecs;

bool DeviceRuntime<SERIAL>::SyncAllKernelsAndCheckErrors = false;
bool MemoryManager<SERIAL>::ReduceMemoryFootprint = false;
bool DeviceRuntime<SERIAL>::TimingAllKernels = false;
bool DeviceRuntime<SERIAL>::PrintKernelConfig = false;

template <> bool deviceAvailable<SERIAL>() {
  return DeviceRuntime<SERIAL>::GetDeviceCount() > 0;
}

} // namespace mgard_x