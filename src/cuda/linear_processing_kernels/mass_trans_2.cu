/* 
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/linear_processing_kernel.h"
#include "cuda/linear_processing_kernel.hpp"

namespace mgard_cuda {

#define KERNELS(T, D) \
                        template void lpk_reo_2<T, D>(\
                        mgard_cuda_handle<T, D> &handle,\
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

  KERNELS(double, 1)
  KERNELS(float,  1)
  KERNELS(double, 2)
  KERNELS(float,  2)
  KERNELS(double, 3)
  KERNELS(float,  3)
  KERNELS(double, 4)
  KERNELS(float,  4)
  KERNELS(double, 5)
  KERNELS(float,  5)

#undef KERNELS

#define KERNELS(T, D) \
                        template void lpk_reo_2<T, D>(\
                        mgard_cuda_handle<T, D> &handle,\
                        int * shape_h, int * shape_c_h, int * shape_d, int * shape_c_d,\
                        int * ldvs, int * ldws,\
                        int processed_n, int * processed_dims_h, int * processed_dims_d,\
                        int curr_dim_r, int curr_dim_c, int curr_dim_f,\
                        T *ddist_c, T *dratio_c,\
                        T *dv1, int lddv11, int lddv12,\
                        T *dv2, int lddv21, int lddv22,\
                        T *dw, int lddw1, int lddw2,\
                        int queue_idx, int config);

  KERNELS(double, 1)
  KERNELS(float,  1)
  KERNELS(double, 2)
  KERNELS(float,  2)
  KERNELS(double, 3)
  KERNELS(float,  3)
  KERNELS(double, 4)
  KERNELS(float,  4)
  KERNELS(double, 5)
  KERNELS(float,  5)

#undef KERNELS

} //end namespace