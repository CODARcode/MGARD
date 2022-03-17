/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

int DeviceRuntime<Serial>::curr_dev_id = 0;
DeviceQueues<Serial> DeviceRuntime<Serial>::queues;
DeviceSpecification<Serial> DeviceRuntime<Serial>::DeviceSpecs;

bool DeviceRuntime<Serial>::SyncAllKernelsAndCheckErrors = false;
bool MemoryManager<Serial>::ReduceMemoryFootprint = false;
bool DeviceRuntime<Serial>::TimingAllKernels = false;
bool DeviceRuntime<Serial>::PrintKernelConfig = false;

AutoTuningTable<Serial> AutoTuner<Serial>::autoTuningTable;
bool AutoTuner<Serial>::ProfileKernels = false;

template <> bool deviceAvailable<Serial>() {
  return DeviceRuntime<Serial>::GetDeviceCount() > 0;
}

} // namespace mgard_x