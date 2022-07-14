/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "mgard-x/RuntimeX/RuntimeXPublic.h"
#include "mgard-x/Utilities/Types.h"
#include <cstdint>

#ifndef MGARD_X_API_H
#define MGARD_X_API_H

namespace mgard_x {

//!\file
//!\brief High level compression and decompression API.

//! Compress a function on an N-D tensor product grid with uniform spacing
//!
//!\param[in] D Dimension.
//!\param[in] dtype Data type Float or Double
//!\param[in] shape Shape of the Dataset to be compressed
//!\param[in] tol Error tolerance.
//!\param[in] s Smoothness parameter.
//!\param[in] mode Error bound type REL or ABS.
//!\param[in] original_data Dataset to be compressed.
//!\param[out] compressed_data Compressed data.
//!\param[out] compressed_size Size of compressed data.
//!\param[in] output_pre_allocated Indicate whether the output buffer is
//! pre-allocated or not.
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              bool output_pre_allocated);

//!\file
//!\brief High level compression and decompression API.

//! Compress a function on an N-D tensor product grid with uniform spacing
//!
//!\param[in] D Dimension.
//!\param[in] dtype Data type Float or Double
//!\param[in] shape Shape of the Dataset to be compressed
//!\param[in] tol Error tolerance.
//!\param[in] s Smoothness parameter.
//!\param[in] mode Error bound type REL or ABS.
//!\param[in] original_data Dataset to be compressed.
//!\param[out] compressed_data Compressed data.
//!\param[out] compressed_size Size of compressed data.
//!\param[in] config For configuring the compression process.
//!\param[in] output_pre_allocated Indicate whether the output buffer is
//! pre-allocated or not.
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size, Config config,
              bool output_pre_allocated);

//! Compress a function on an N-D tensor product grid with non-uniform spacing
//!
//!\param[in] D Dimension.
//!\param[in] dtype Data type Float or Double
//!\param[in] shape Shape of the Dataset to be compressed
//!\param[in] tol Error tolerance.
//!\param[in] s Smoothness parameter.
//!\param[in] mode Error bound type REL or ABS.
//!\param[in] original_data Dataset to be compressed.
//!\param[out] compressed_data Compressed data.
//!\param[out] compressed_size Size of comrpessed data.
//!\param[in] coords Coordinates data.
//!\param[in] output_pre_allocated Indicate whether the output buffer is
//! pre-allocated or not.
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, bool output_pre_allocated);

//! Compress a function on an N-D tensor product grid with non-uniform spacing
//!
//!\param[in] D Dimension.
//!\param[in] dtype Data type Float or Double
//!\param[in] shape Shape of the Dataset to be compressed
//!\param[in] tol Error tolerance.
//!\param[in] s Smoothness parameter.
//!\param[in] mode Error bound type REL or ABS.
//!\param[in] original_data Dataset to be compressed.
//!\param[out] compressed_data Compressed data.
//!\param[out] compressed_size Size of comrpessed data.
//!\param[in] coords Coordinates data.
//!\param[in] config For configuring the compression process.
//!\param[in] output_pre_allocated Indicate whether the output buffer is
//! pre-allocated or not.
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, Config config,
              bool output_pre_allocated);

//! Decompress a function on an N-D tensor product grid
//!
//!\param[in] compressed_data Compressed data.
//!\param[in] compressed_size Size of comrpessed data.
//!\param[out] decompressed_data Decompressed data.
//!\param[in] output_pre_allocated Indicate whether the output buffer is
//! pre-allocated or not.
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, bool output_pre_allocated);

//! Decompress a function on an N-D tensor product grid
//!
//!\param[in] compressed_data Compressed data.
//!\param[in] compressed_size Size of comrpessed data.
//!\param[out] decompressed_data Decompressed data.
//!\param[in] config For configuring the decompression process.
//!\param[in] output_pre_allocated Indicate whether the output buffer is
//! pre-allocated or not.
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, Config config,
                bool output_pre_allocated);

//! Decompress a function on an N-D tensor product grid
//!
//!\param[in] compressed_data Compressed data.
//!\param[in] compressed_size Size of comrpessed data.
//!\param[out] decompressed_data Decompressed data.
//!\param[out] shape Shape of decompressed data.
//!\param[out] dtype Data type of decompressed data.
//!\param[in] config For configuring the decompression process.
//!\param[in] output_pre_allocated Indicate whether the output buffer is
//! pre-allocated or not.
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, std::vector<mgard_x::SIZE> &shape,
                data_type &dtype, Config config, bool output_pre_allocated);

//! Decompress a function on an N-D tensor product grid
//!
//!\param[in] compressed_data Compressed data.
//!\param[in] compressed_size Size of comrpessed data.
//!\param[out] decompressed_data Decompressed data.
//!\param[out] shape Shape of decompressed data.
//!\param[out] dtype Data type of decompressed data.
//!\param[in] output_pre_allocated Indicate whether the output buffer is
//! pre-allocated or not.
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, std::vector<mgard_x::SIZE> &shape,
                data_type &dtype, bool output_pre_allocated);

} // namespace mgard_x

#endif
