/* 
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <assert.h>
#include <iostream>

#include "cuda/mgard_cuda_compression_workflow.h"
#include "cuda/mgard_cuda_common.h"

namespace mgard_cuda {

template <typename T, int D>
unsigned char *compress(mgard_cuda_handle<T, D> &handle, T *v, size_t &out_size, T tol, T s)
// Perform compression preserving the tolerance in the L-infty norm
{
  assert(tol >= 1e-7);
  unsigned char *mgard_compressed_ptr = mgard_cuda::refactor_qz_cuda<T, D>(handle, v, out_size, tol, s);
  return mgard_compressed_ptr;
}

template <typename T, int D>
T *decompress(mgard_cuda_handle<T, D> &handle, unsigned char *data,
                         size_t data_len) {
  T *mgard_decompressed_ptr = mgard_cuda::recompose_udq_cuda<T, D>(handle, data, data_len);
  return mgard_decompressed_ptr;
}


#define API(T, D) \
        template unsigned char * compress<T, D>(\
        mgard_cuda_handle<T, D> &handle,\
        T *v, size_t &out_size, T tol, T s);\
        template T * decompress<T, D>(\
        mgard_cuda_handle<T, D> &handle,\
        unsigned char *data, size_t data_len);

API(double, 1)
API(float,  1)
API(double, 2)
API(float,  2)
API(double, 3)
API(float,  3)
API(double, 4)
API(float,  4)
API(double, 5)
API(float,  5)

#undef API

} // end namespace