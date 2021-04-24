/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <assert.h>
#include <iostream>

#include "cuda/Common.h"
#include "cuda/CompressionWorkflow.h"

namespace mgard_cuda {

template <typename T, int D>
Array<unsigned char, 1> compress(Handle<T, D> &handle, Array<T, D> &in_array,
                                 enum error_bound_type type, T tol, T s)
// Perform compression preserving the tolerance in the L-infty norm
{
  assert(tol >= 1e-7);
  return mgard_cuda::refactor_qz_cuda<T, D>(handle, in_array, type, tol, s);
}

template <typename T, int D>
Array<T, D> decompress(Handle<T, D> &handle,
                       Array<unsigned char, 1> &compressed_array) {
  return mgard_cuda::recompose_udq_cuda<T, D>(handle, compressed_array);
}

#define API(T, D)                                                              \
  template Array<unsigned char, 1> compress<T, D>(                             \
      Handle<T, D> & handle, Array<T, D> & in_array,                           \
      enum error_bound_type type, T tol, T s);                                 \
  template Array<T, D> decompress<T, D>(                                       \
      Handle<T, D> & handle, Array<unsigned char, 1> & compressed_array);

API(double, 1)
API(float, 1)
API(double, 2)
API(float, 2)
API(double, 3)
API(float, 3)
API(double, 4)
API(float, 4)
API(double, 5)
API(float, 5)

#undef API

} // namespace mgard_cuda