/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda_helper.h"

namespace mgard_cuda {

template <typename T, int D>
unsigned char *refactor_qz_cuda(mgard_cuda_handle<T, D> &handle, T *u,
                                size_t &outsize, T tol, T s);

template <typename T, int D>
T *recompose_udq_cuda(mgard_cuda_handle<T, D> &handle, unsigned char *data,
                      size_t data_len);

} // namespace mgard_cuda
