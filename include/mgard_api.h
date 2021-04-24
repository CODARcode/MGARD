// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney, Qing Liu, Jieyang Chen
// Corresponding Author: Ben Whitney, Qing Liu, Jieyang Chen
//
// version: 0.1.0
// See LICENSE for details.
#ifndef MGARD_API_H
#define MGARD_API_H
//!\file
//!\brief Compression and decompression API.

#include "TensorMeshHierarchy.hpp"

#include <memory>

#include "cuda/Common.h"
#include "cuda/CompressionWorkflow.h"
#include "cuda/MemoryManagement.h"

//! Implementation of the MGARD compression and decompression algorithms.
namespace mgard {

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
  void const *data() const;

  //! Return the size in bytes of the compressed dataset.
  std::size_t size() const;

private:
  //! Compressed dataset.
  std::unique_ptr<const unsigned char[]> data_;

  //! Size of the compressed dataset in bytes.
  const std::size_t size_;
};

//! Decompressed dataset and associated compression parameters.
template <std::size_t N, typename Real> class DecompressedDataset {
public:
  //! Constructor.
  //!
  //! The buffer pointed to by `data` is freed when this object is destructed.
  //! It should be allocated with `new unsigned char[size]`.
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
template <typename T, int D>
Array<unsigned char, 1> compress(Handle<T, D> &handle, Array<T, D> &in_array,
                                 enum error_bound_type type, T tol, T s);

//! Decompress a function on an N-D tensor product grid
//!
//!\param[in] handle Handle type for storing precomputed variable to
//! help speed up decompression.
//!\param[in] compressed_array Compressed dataset.
//!\return Decompressed dataset.
template <typename T, int D>
Array<T, D> decompress(Handle<T, D> &handle,
                       Array<unsigned char, 1> &compressed_array);

} // namespace mgard_cuda

#include "mgard_api.tpp"
#endif
