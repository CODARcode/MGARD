/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/IterativeProcessingKernel.h"
#include "cuda/IterativeProcessingKernel.hpp"

namespace mgard_cuda {

#define KERNELS(T, D)                                                          \
  template void ipk_3<T, D>(\
                    Handle<T, D> &handle,\
                    thrust::device_vector<int> shape,\
                    thrust::device_vector<int> shape_c,\ 
                    thrust::device_vector<int> ldvs,\ 
                    thrust::device_vector<int> ldws,\
                    thrust::device_vector<int> processed_dims,\
                    int curr_dim_r, int curr_dim_c, int curr_dim_f,\
                    T * am, T *bm,\
                    T * ddist_r, T *dv,\
                    int lddv1, int lddv2, int queue_idx, int config);

KERNELS(double, 1)
KERNELS(float, 1)
KERNELS(double, 2)
KERNELS(float, 2)
KERNELS(double, 3)
KERNELS(float, 3)
KERNELS(double, 4)
KERNELS(float, 4)
KERNELS(double, 5)
KERNELS(float, 5)
#undef KERNELS

#define KERNELS(T, D)                                                          \
  template void ipk_3<T, D>(                                                   \
      Handle<T, D> & handle, int *shape_h, int *shape_c_h, int *shape_d,       \
      int *shape_c_d, int *ldvs, int *ldws, int processed_n,                   \
      int *processed_dims_h, int *processed_dims_d, int curr_dim_r,            \
      int curr_dim_c, int curr_dim_f, T *am, T *bm, T *ddist_r, T *dv,         \
      int lddv1, int lddv2, int queue_idx, int config);

KERNELS(double, 1)
KERNELS(float, 1)
KERNELS(double, 2)
KERNELS(float, 2)
KERNELS(double, 3)
KERNELS(float, 3)
KERNELS(double, 4)
KERNELS(float, 4)
KERNELS(double, 5)
KERNELS(float, 5)
#undef KERNELS

} // namespace mgard_cuda