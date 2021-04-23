/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "cuda/Common.h"
#include "cuda/CompressionWorkflow.h"
#include "cuda/MemoryManagement.h"

#ifndef MGARD_API_CUDA_H
#define MGARD_API_CUDA_H

namespace mgard_cuda {

//!\file
//!\brief Compression and decompression API.

//! Compress a function on an equispaced N-D tensor product grid while
//! controlling the error as measured in the \f$ L^{\infty} \f$ norm.
//!
//!\param[in] handle Handle type for storing precomputed variable to
//! help speedup compression. \param[in] data Dataset to be compressed.
//!\param[out] out_size Size in bytes of the compressed dataset.
//!\param[in] tol Relative error tolerance.
//!\param[in] s S-norm.
//!
//!\return Compressed dataset.
template <typename T, int D>
Array<unsigned char, 1> compress(Handle<T, D> &handle, Array<T, D> &in_array,
                                 T tol, T s);

//! Decompress a function on an equispaced N-D tensor product grid which was
//! compressed while controlling the error as measured in the \f$ L^{\infty} \f$
//! norm.
//!
//!\param[in] handle Handle type for storing precomputed variable to
//! help speedup decompression. \param[in] data Compressed dataset. \param[in]
//! data_len Size in bytes of the compressed dataset.
//!
//!\return Decompressed dataset.
template <typename T, int D>
Array<T, D> decompress(Handle<T, D> &handle,
                       Array<unsigned char, 1> &compressed_array);

} // namespace mgard_cuda

#endif