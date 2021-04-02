/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <assert.h>
#include <iostream>

#include "cuda/mgard_cuda_common.h"
#include "cuda/mgard_cuda_compression_workflow.h"

namespace mgard {

template <typename Real, int N>
unsigned char *compress_cuda(mgard_cuda_handle<Real, N> &handle, Real *v,
                             size_t &out_size, Real tol, Real s)
// Perform compression preserving the tolerance in the L-infty norm
{
  assert(tol >= 1e-7);
  unsigned char *mgard_compressed_ptr =
      mgard_cuda::refactor_qz_cuda<Real, N>(handle, v, out_size, tol, s);
  return mgard_compressed_ptr;
}

template <typename Real, int N>
Real *decompress_cuda(mgard_cuda_handle<Real, N> &handle, unsigned char *data,
                      size_t data_len) {
  Real *mgard_decompressed_ptr =
      mgard_cuda::recompose_udq_cuda<Real, N>(handle, data, data_len);
  return mgard_decompressed_ptr;
}

#define API(Real, N)                                                           \
  template unsigned char *compress_cuda<Real, N>(                              \
      mgard_cuda_handle<Real, N> & handle, Real * v, size_t & out_size,        \
      Real tol, Real s);                                                       \
  template Real *decompress_cuda<Real, N>(mgard_cuda_handle<Real, N> & handle, \
                                          unsigned char *data,                 \
                                          size_t data_len);

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

} // namespace mgard