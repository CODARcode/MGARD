// Copyright 2020, Oak Ridge National Laboratory.
// MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by NVIDIA GPUs
// Authors: Jieyang Chen
// Corresponding Author: Jieyang Chen
//
// version: 0.0.0.1
// See LICENSE for details.

#include "cuda/mgard_cuda_common.h"
#include "cuda/mgard_cuda_compression_workflow.h"
#include "cuda/mgard_cuda_helper.h"

#ifndef MGARD_API_CUDA_H
#define MGARD_API_CUDA_H

namespace mgard {
//! Compression with GPU
//! controlling the error as measured in the \f$ L^{\infty} \f$ norm.
//!\param[in] handle mgard_cuda_handle type for storing precomputed variable to
//! help speeding up compression.
//!\param[in] data Dataset to be compressed.
//!\param[out] out_size Size in bytes of the compressed dataset.
//!\param[in] tol Relative error tolerance.
//!\param[in] s Smoothness parameter to use in compressing the function.
//!
//!\return Compressed dataset.
template <typename Real, int N>
unsigned char *compress_cuda(mgard_cuda_handle<Real, N> &handle, Real *v,
                             size_t &out_size, Real tol, Real s);

//! Decompression with GPU
//!\param[in] handle mgard_cuda_handle type for storing precomputed variable to
//! help speeding up decompression.
//!\param[in] data Compressed dataset.
//!\param[in]
//! data_len Size in bytes of the compressed dataset.
//!
//!\return Decompressed dataset.
template <typename Real, int N>
Real *decompress_cuda(mgard_cuda_handle<Real, N> &handle, unsigned char *data,
                      size_t data_len);

} // namespace mgard
#endif