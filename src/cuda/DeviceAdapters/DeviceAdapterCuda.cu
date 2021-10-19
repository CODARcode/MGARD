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
bool DeviceRuntime<CUDA>::SyncAllKernelsAndCheckErrors = false;
bool MemoryManager<CUDA>::ReduceMemoryFootprint = false;

}
#endif