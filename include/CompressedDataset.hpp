#ifndef COMPRESSED_DATASET_HPP
#define COMPRESSED_DATASET_HPP
//!\file
//!\brief Compressed and decompressed datasets and metadata.

#include <cstddef>

#include <memory>
#include <ostream>

#include "proto/mgard.pb.h"

#include "TensorMeshHierarchy.hpp"

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
  //!\param header Compressed dataset header.
  //!\param s Smoothness parameter.
  //!\param tolerance Error tolerance.
  //!\param data Compressed dataset.
  //!\param size Size of the compressed dataset in bytes.
  CompressedDataset(const TensorMeshHierarchy<N, Real> &hierarchy,
                    const pb::Header &header, const Real s,
                    const Real tolerance, void const *const data,
                    const std::size_t size);

  //! Mesh hierarchy used in compressing the dataset.
  const TensorMeshHierarchy<N, Real> hierarchy;

  //! Header for compressed dataset.
  //!
  //! *This is an experimental part of the API.*
  pb::Header header;

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

  //! Serialize the compressed dataset.
  //!
  //! *This is an experimental part of the API.*
  void write(std::ostream &ostream) const;

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
  //! It should be allocated with `new Real[size]`.
  //!
  //!\param compressed Compressed dataset which was decompressed.
  //!\param data Nodal values of the decompressed function.
  DecompressedDataset(const CompressedDataset<N, Real> &compressed,
                      Real const *const data);

  //! Mesh hierarchy used in compressing the original dataset.
  const TensorMeshHierarchy<N, Real> hierarchy;

  //! Header for decompressed dataset.
  //!
  //! *This is an experimental part of the API.*
  pb::Header header;

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

} // namespace mgard

#include "CompressedDataset.tpp"
#endif
