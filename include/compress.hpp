// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney, Qing Liu, Jieyang Chen
// Corresponding Author: Ben Whitney, Qing Liu, Jieyang Chen
//
// See LICENSE for details.
#ifndef COMPRESS_HPP
#define COMPRESS_HPP
//!\file
//!\brief Compression and decompression API.

#include "CompressedDataset.hpp"
#include "TensorMeshHierarchy.hpp"
#include "utilities.hpp"

#ifdef __NVCC__
#error "Please include `compress_cuda.hpp` instead of `compress.hpp` when "\
  "compiling with NVCC."
#endif

#include "compress_cuda.hpp"

//! Implementation of the MGARD compression and decompression algorithms.
namespace mgard {

//! Compress a function on a tensor product grid.
//!
//!\param hierarchy Mesh hierarchy to use in compressing the function.
//!\param v Nodal values of the function.
//!\param s Smoothness parameter to use in compressing the function.
//!\param tolerance Absolute error tolerance to use in compressing the function.
template <std::size_t N, typename Real>
CompressedDataset<N, Real>
compress(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v,
         const Real s, const Real tolerance);

//! Decompress a function on a tensor product grid.
//!
//!\param compressed Compressed function to be decompressed.
template <std::size_t N, typename Real>
DecompressedDataset<N, Real>
decompress(const CompressedDataset<N, Real> &compressed);

//! Decompress a dataset stored in self-describing format.
//!
//! *This is an experimental part of the API.*
//!
//!\param data Self-describing compressed dataset.
//!\param size Size in bytes of compressed dataset.
MemoryBuffer<const unsigned char> decompress(void const *const data,
                                             const std::size_t size);

} // namespace mgard

namespace mgard_x {

//!\file
//!\brief Compression and decompression API.

//! Compress a function on an N-D tensor product grid
//!
//!\param[in] handle Handle type for storing precomputed variable to
//! help speed up compression.
//!\param[in] in_array Dataset to be compressed.
//!\param[in] type Error bound type: REL or ABS.
//!\param[in] tol Relative error tolerance.
//!\param[in] s Smoothness parameter to use in compressing the function.
//!
//!\return Compressed dataset.
template <uint32_t D, typename T, typename DeviceType>
Array<1, unsigned char, DeviceType> compress(Handle<D, T, DeviceType> &handle, Array<D, T, DeviceType> &in_array,
                                 enum error_bound_type type, T tol, T s);

//! Decompress a function on an N-D tensor product grid
//!
//!\param[in] handle Handle type for storing precomputed variable to
//! help speed up decompression.
//!\param[in] compressed_array Compressed dataset.
//!\return Decompressed dataset.
template <uint32_t D, typename T, typename DeviceType>
Array<D, T, DeviceType> decompress(Handle<D, T, DeviceType> &handle,
                       Array<1, unsigned char, DeviceType> &compressed_array);

} // namespace mgard_x

#include "compress.tpp"
#endif
