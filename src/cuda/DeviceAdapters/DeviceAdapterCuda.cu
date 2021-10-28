/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGARD_CUDA_DEVICE_ADAPTER_CUDA_CU
#define MGARD_CUDA_DEVICE_ADAPTER_CUDA_CU

#include "cuda/CommonInternal.h"

#include "cuda/DeviceAdapters/DeviceAdapterCuda.h"

namespace mgard_cuda {

int DeviceRuntime<CUDA>::curr_dev_id = 0;
DeviceQueues<CUDA> DeviceRuntime<CUDA>::queues;
DeviceSpecification<CUDA> DeviceRuntime<CUDA>::DeviceSpecs;

bool DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors = true;
bool MemoryManager<CUDA>::ReduceMemoryFootprint = false;

KernelConfigs<CUDA> AutoTuner<CUDA>::kernelConfigs;
AutoTuningTable<CUDA> AutoTuner<CUDA>::autoTuningTable;
bool AutoTuner<CUDA>::ProfileKernels = false;

}
#endif