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
#error "Please include `compress_x.hpp` instead of `compress.hpp` when "\
  "compiling with NVCC."
#endif

#include "compress_x.hpp"

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

#include "compress.tpp"
#endif
