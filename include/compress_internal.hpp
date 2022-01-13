#ifndef COMPRESS_INTERNAL_HPP
#define COMPRESS_INTERNAL_HPP
//!\file
//!\brief Intermediate functions for decompressing self-describing buffers.

#include <cstddef>

#include <memory>

#include "proto/mgard.pb.h"

#include "TensorMeshHierarchy.hpp"

namespace mgard {

//! Decompress a dataset originally stored in self-describing format.
//!
//! *This function is not part of the API.*
//!
//! This function is called once the header has been parsed. Its job is to
//! determine the topology dimension.
//!
//!\param header Header parsed from the original self-describing buffer.
//!\param data Compressed buffer. Note that this parameter must not point to the
//! start of the header.
//!\param size Size in bytes of compressed buffer. Note that this parameter must
//! not include the size of the header.
MemoryBuffer<const unsigned char> decompress(const pb::Header &header,
                                             void const *const data,
                                             const std::size_t size);

//! Decompress a dataset originally stored in self-describing format.
//!
//! *This function is not part of the API.*
//!
//! This function is called once the topology dimension is determined. Its job
//! to call the appropriate decompression function template (with the second
//! argument as the template parameter).
//!
//!\param header Header parsed from the original self-describing buffer.
//!\param dimension Dimension of the mesh on which the dataset is defined.
//!\param data Compressed buffer. Note that this parameter must not point to the
//! start of the header.
//!\param size Size in bytes of compressed buffer. Note that this parameter must
//! not include the size of the header.
MemoryBuffer<const unsigned char> decompress(const pb::Header &header,
                                             const std::size_t dimension,
                                             void const *const data,
                                             const std::size_t size);

// Doxygen threw an error when this function was named `decompress`.

//! Decompress a dataset originally stored in self-describing format.
//!
//! *This function is not part of the API.*
//!
//! This function is called once the topology dimension has been determined from
//! the header. Its job is to determine the data type.
//!
//!\param header Header parsed from the original self-describing buffer.
//!\param data Compressed buffer. Note that this parameter must not point to the
//! start of the header.
//!\param size Size in bytes of compressed buffer. Note that this parameter must
//! not include the size of the header.
template <std::size_t N>
MemoryBuffer<const unsigned char> decompress_N(const pb::Header &header,
                                               void const *const data,
                                               const std::size_t size);

// Doxygen threw an error when this function was named `decompress`.

//! Decompress a dataset originally stored in self-describing format.
//!
//! *This function is not part of the API.*
//!
//! This function is called once the topology dimension and data type have been
//! determined from the header. Its job is to create the `TensorMeshHierarchy`.
//!
//!\param header Header parsed from the original self-describing buffer.
//!\param data Compressed buffer. Note that this parameter must not point to the
//! start of the header.
//!\param size Size in bytes of compressed buffer. Note that this parameter must
//! not include the size of the header.
template <std::size_t N, typename Real>
MemoryBuffer<const unsigned char> decompress_N_Real(const pb::Header &header,
                                                    void const *const data,
                                                    const std::size_t size);

//! Decompress a dataset originally stored in self-describing format.
//!
//! *This function is not part of the API.*
//!
//! This function is called once the `TensorMeshHierarchy` has been created. Its
//! job is to call the `decompress` overload taking a `CompressedDataset`.
//!
//!\param hierarchy Mesh hierarchy used in compressing the dataset.
//!\param header Header parsed from the original self-describing buffer.
//!\param data Compressed buffer. Note that this parameter must not point to the
//! start of the header.
//!\param size Size in bytes of compressed buffer. Note that this parameter must
//! not include the size of the header.
template <std::size_t N, typename Real>
MemoryBuffer<const unsigned char>
decompress(const TensorMeshHierarchy<N, Real> &hierarchy,
           const pb::Header &header, void const *const data,
           const std::size_t size);

} // namespace mgard

#include "compress_internal.tpp"
#endif
