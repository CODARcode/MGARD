/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_DEVICE_ADAPTER_SERIAL_CPP
#define MGARD_X_DEVICE_ADAPTER_SERIAL_CPP

#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

int DeviceRuntime<Serial>::curr_dev_id = 0;
DeviceQueues<Serial> DeviceRuntime<Serial>::queues;
DeviceSpecification<Serial> DeviceRuntime<Serial>::DeviceSpecs;

bool DeviceRuntime<Serial>::SyncAllKernelsAndCheckErrors = false;
bool MemoryManager<Serial>::ReduceMemoryFootprint = false;

KernelConfigs<Serial> AutoTuner<Serial>::kernelConfigs;
AutoTuningTable<Serial> AutoTuner<Serial>::autoTuningTable;
bool AutoTuner<Serial>::ProfileKernels = false;

}
#endif