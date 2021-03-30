// Copyright 2020, Oak Ridge National Laboratory.
// MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by NVIDIA GPUs
// Authors: Jieyang Chen
// Corresponding Author: Jieyang Chen
//
// version: 0.0.0.1
// See LICENSE for details.

#include "cuda/mgard_cuda_compression_workflow.h"
#include "cuda/mgard_cuda_common.h"
#include "cuda/mgard_cuda_helper.h"

#ifndef MGARD_API_CUDA_H
#define MGARD_API_CUDA_H
//!\file
//!\brief Compression and decompression API.

//! Compress a function on an equispaced 3D tensor product grid while
//! controlling the error as measured in the \f$ L^{\infty} \f$ norm.
//!
//!\param[in] handle mgard_cuda_handle type for storing precomputed variable to
//! help speedup compression. \param[in] data Dataset to be compressed.
//!\param[out] out_size Size in bytes of the compressed dataset.
//!\param[in] tol Relative error tolerance.
//!
//!\return Compressed dataset.
template <typename T, int D>
unsigned char *mgard_compress_cuda(mgard_cuda_handle<T, D> &handle, T *v,
                                   size_t &out_size, T tol, T s);

//! Decompress a function on an equispaced 3D tensor product grid which was
//! compressed while controlling the error as measured in the \f$ L^{\infty} \f$
//! norm.
//!
//!\param[in] handle mgard_cuda_handle type for storing precomputed variable to
//! help speedup decompression. \param[in] data Compressed dataset. \param[in]
//! data_len Size in bytes of the compressed dataset.
//!
//!\return Decompressed dataset.
template <typename T, int D>
T *mgard_decompress_cuda(mgard_cuda_handle<T, D> &handle, unsigned char *data,
                         size_t data_len);

#endif