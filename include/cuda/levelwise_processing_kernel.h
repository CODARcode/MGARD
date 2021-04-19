/* 
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */


#ifndef MGRAD_CUDA_LEVELWISE_PROCESSING_KERNEL
#define MGRAD_CUDA_LEVELWISE_PROCESSING_KERNEL

#include "mgard_cuda_common.h"
#include "mgard_cuda_common_internal.h"

namespace mgard_cuda {

template <typename T, int D, int OP>
void lwpk(mgard_cuda_handle<T, D> &handle, thrust::device_vector<int> shape,
                   T *dv, thrust::device_vector<int> ldvs, T *dwork, thrust::device_vector<int> ldws,
                   int queue_idx);

template <typename T, int D, int OP>
void lwpk(mgard_cuda_handle<T, D> &handle, 
          int * shape_h, int * shape_d,
          T *dv, int * ldvs, 
          T *dwork, int * ldws,
          int queue_idx);

}

#endif