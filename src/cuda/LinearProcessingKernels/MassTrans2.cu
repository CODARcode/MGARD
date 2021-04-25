/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/LinearProcessingKernel.h"
#include "cuda/LinearProcessingKernel.hpp"

namespace mgard_cuda {

#define KERNELS(D, T)                                                          \
  template void lpk_reo_2<D, T>(\
                        Handle<D, T> &handle,\
                        thrust::device_vector<int> shape,\
                        thrust::device_vector<int> shape_c,\ 
                        thrust::device_vector<int> ldvs,\ 
                        thrust::device_vector<int> ldws,\
                        thrust::device_vector<int> processed_dims,\
                        int curr_dim_r, int curr_dim_c, int curr_dim_f,\
                        T *ddist_c, T *dratio_c,\
                        T *dv1, int lddv11, int lddv12,\
                        T *dv2, int lddv21, int lddv22,\
                        T *dw, int lddw1, int lddw2,\
                        int queue_idx, int config);

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)

#undef KERNELS

#define KERNELS(D, T)                                                          \
  template void lpk_reo_2<D, T>(                                               \
      Handle<D, T> & handle, int *shape_h, int *shape_c_h, int *shape_d,       \
      int *shape_c_d, int *ldvs, int *ldws, int processed_n,                   \
      int *processed_dims_h, int *processed_dims_d, int curr_dim_r,            \
      int curr_dim_c, int curr_dim_f, T *ddist_c, T *dratio_c, T *dv1,         \
      int lddv11, int lddv12, T *dv2, int lddv21, int lddv22, T *dw,           \
      int lddw1, int lddw2, int queue_idx, int config);

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)

#undef KERNELS

} // namespace mgard_cuda