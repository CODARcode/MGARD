/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "mgard-x-160/RuntimeX/RuntimeX.h"

namespace mgard_x_160 {

int serial_dev_id = 0;
DeviceQueues<SERIAL> DeviceRuntime<SERIAL>::queues;
DeviceSpecification<SERIAL> DeviceRuntime<SERIAL>::DeviceSpecs;

bool DeviceRuntime<SERIAL>::SyncAllKernelsAndCheckErrors = false;
bool DeviceRuntime<SERIAL>::TimingAllKernels = false;
bool DeviceRuntime<SERIAL>::PrintKernelConfig = false;

template <> bool deviceAvailable<SERIAL>() {
  return DeviceRuntime<SERIAL>::GetDeviceCount() > 0;
}

} // namespace mgard_x_160