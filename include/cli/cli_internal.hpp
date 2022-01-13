#ifndef CLI_INTERNAL_HPP
#define CLI_INTERNAL_HPP
//!\file
//!\brief Intermediate functions for compressing untyped buffers.

#include <cstddef>

#include "cli/arguments.hpp"

namespace cli {

//! Compress a dataset contained in a file.
//!
//! This function is called once the command line arguments have been parsed.
//! Its job is to determine the topology dimension.
//!
//!\param arguments Command line arguments passed to the compression script.
int compress(const CompressionArguments &arguments);

//! Compress a dataset contained in a file.
//!
//! This function is called once the topology dimension has been determined. Its
//! job is to call the appropriate compression function template (with the
//! second argument as the template parameter).
//!
//!\param arguments Command line arguments passed to the compression script.
//!\param dimension Dimension of the mesh on which the dataset is defined.
int compress(const CompressionArguments &arguments,
             const std::size_t dimension);

//! Compress a dataset contained in a file.
//!
//! This function is called once the topology dimension has been determined. Its
//! job is to determine the data type.
//!
//!\param arguments Command line arguments passed to the compression script.
template <std::size_t N> int compress_N(const CompressionArguments &arguments);

//! Compress a dataset contained in a file.
//!
//! This function is called once the topology dimension and data type have been
//! determined. Its job is to create the `TensorMeshHierarchy` and call
//! `mgard::compress`.
//!
//!\param arguments Command line arguments passed to the compression script.
template <std::size_t N, typename Real>
int compress_N_Real(const CompressionArguments &arguments);

//! Decompressed a dataset contained in a file.
//!
//!\param arguments Command line arguments passed to the decompression script.
int decompress(const DecompressionArguments &arguments);

} // namespace cli

#include "cli/cli_internal.tpp"
#endif
