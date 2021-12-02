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

#include "TensorMeshHierarchy.hpp"

#include <array>
#include <memory>

#ifdef __NVCC__
#error "Please include `compress_cuda.hpp` instead of `compress.hpp` when "\
  "compiling with NVCC."
#endif

#include "compress_cuda.hpp"

#ifdef MGARD_PROTOBUF
#include <ostream>

#include "proto/mgard.pb.h"
#endif

//! Implementation of the MGARD compression and decompression algorithms.
namespace mgard {

//! Magic bytes for MGARD file format ('MGARD' in ASCII).
inline constexpr std::array<char, 5> SIGNATURE{0x4d, 0x47, 0x41, 0x52, 0x44};

//! Compressed dataset and associated compression parameters.
template <std::size_t N, typename Real> class CompressedDataset {
public:
  //! Constructor.
  //!
  //! The buffer pointed to by `data` is freed when this object is destructed.
  //! It should be allocated with `new unsigned char[size]`.
  //!
  //!\param hierarchy Associated mesh hierarchy.
  //!\param s Smoothness parameter.
  //!\param tolerance Error tolerance.
  //!\param data Compressed dataset.
  //!\param size Size of the compressed dataset in bytes.
  CompressedDataset(const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
                    const Real tolerance, void const *const data,
                    const std::size_t size);

  //! Mesh hierarchy used in compressing the dataset.
  const TensorMeshHierarchy<N, Real> hierarchy;

  //! Smoothness parameter used in compressing the dataset.
  const Real s;

  //! Error tolerance used in compressing the dataset.
  const Real tolerance;

  //! Return a pointer to the compressed dataset.
  //!
  //! *The format of the compressed dataset is an experimental part of the API.*
  void const *data() const;

  //! Return the size in bytes of the compressed dataset.
  std::size_t size() const;

#ifdef MGARD_PROTOBUF
  //! Return a pointer to the compressed dataset header.
  //!
  //! *This is an experimental part of the API.*
  pb::Header const *header() const;

  //! Serialize the compressed dataset.
  //!
  //! *This is an experimental part of the API.*
  void write(std::ostream &ostream) const;
#endif

private:
  //! Compressed dataset.
  std::unique_ptr<const unsigned char[]> data_;

  //! Size of the compressed dataset in bytes.
  const std::size_t size_;

#ifdef MGARD_PROTOBUF
  //! Header for compressed dataset.
  //!
  //! *This is an experimental part of the API.*
  pb::Header header_;
#endif
};

//! Decompressed dataset and associated compression parameters.
template <std::size_t N, typename Real> class DecompressedDataset {
public:
  //! Constructor.
  //!
  //! The buffer pointed to by `data` is freed when this object is destructed.
  //! It should be allocated with `new Real[size]`.
  //!
  //!\param compressed Compressed dataset which was decompressed.
  //!\param data Nodal values of the decompressed function.
  DecompressedDataset(const CompressedDataset<N, Real> &compressed,
                      Real const *const data);

  //! Mesh hierarchy used in compressing the original dataset.
  const TensorMeshHierarchy<N, Real> hierarchy;

  //! Smoothness parameter used in compressing the original dataset.
  const Real s;

  //! Error tolerance used in compressing the original dataset.
  const Real tolerance;

  //! Return a pointer to the decompressed dataset.
  Real const *data() const;

private:
  //! Decompressed dataset.
  std::unique_ptr<const Real[]> data_;

#ifdef MGARD_PROTOBUF
  //! Header for decompressed dataset.
  //!
  //! *This is an experimental part of the API.*
  pb::Header header_;
#endif
};

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
std::unique_ptr<unsigned char const[]> decompress(void const *const data,
                                                  const std::size_t size);

} // namespace mgard

namespace mgard_cuda {

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
template <uint32_t D, typename T>
Array<1, unsigned char> compress(Handle<D, T> &handle, Array<D, T> &in_array,
                                 enum error_bound_type type, T tol, T s);

//! Decompress a function on an N-D tensor product grid
//!
//!\param[in] handle Handle type for storing precomputed variable to
//! help speed up decompression.
//!\param[in] compressed_array Compressed dataset.
//!\return Decompressed dataset.
template <uint32_t D, typename T>
Array<D, T> decompress(Handle<D, T> &handle,
                       Array<1, unsigned char> &compressed_array);

} // namespace mgard_cuda

#include "compress.tpp"
#endif
