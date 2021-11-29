/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_DEVICE_ADAPTER_KOKKOS_CPP
#define MGARD_X_DEVICE_ADAPTER_KOKKOS_CPP

#include "mgard-x/RuntimeX/RuntimeX.h"
 
namespace mgard_x {

int DeviceRuntime<KOKKOS>::curr_dev_id = 0;
// DeviceQueues<CUDA> DeviceRuntime<CUDA>::queues;
DeviceSpecification<KOKKOS> DeviceRuntime<KOKKOS>::DeviceSpecs;

bool DeviceRuntime<KOKKOS>::SyncAllKernelsAndCheckErrors = false;
// bool MemoryManager<CUDA>::ReduceMemoryFootprint = false;

// KernelConfigs<CUDA> AutoTuner<CUDA>::kernelConfigs;
// AutoTuningTable<CUDA> AutoTuner<CUDA>::autoTuningTable;
// bool AutoTuner<CUDA>::ProfileKernels = false;

}
#endif