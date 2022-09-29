/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "mgard-x/RuntimeX/RuntimeX.h"

namespace mgard_x {

int openmp_dev_id = 0;
DeviceQueues<OPENMP> DeviceRuntime<OPENMP>::queues;
DeviceSpecification<OPENMP> DeviceRuntime<OPENMP>::DeviceSpecs;

bool DeviceRuntime<OPENMP>::SyncAllKernelsAndCheckErrors = false;
bool MemoryManager<OPENMP>::ReduceMemoryFootprint = false;
bool DeviceRuntime<OPENMP>::TimingAllKernels = false;
bool DeviceRuntime<OPENMP>::PrintKernelConfig = false;

AutoTuningTable<OPENMP> AutoTuner<OPENMP>::autoTuningTable;
bool AutoTuner<OPENMP>::ProfileKernels = false;

template <> bool deviceAvailable<OPENMP>() {
  return DeviceRuntime<OPENMP>::GetDeviceCount() > 0;
}

} // namespace mgard_x