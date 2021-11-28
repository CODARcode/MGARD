/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "mgard-x/Types.h"
#include "mgard-x/RuntimeX/RuntimeXPublic.h"
#include "mgard-x/CompressionWorkflow.h"
#include <cstdint>

#ifndef MGARD_API_CUDA_H
#define MGARD_API_CUDA_H

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
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size);

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
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size, Config config);

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
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords);

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
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, Config config);

//! Decompress a function on an N-D tensor product grid
//!
//!\param[in] compressed_data Compressed data.
//!\param[in] compressed_size Size of comrpessed data.
//!\param[out] decompressed_data Decompressed data.
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data);

//! Decompress a function on an N-D tensor product grid
//!
//!\param[in] compressed_data Compressed data.
//!\param[in] compressed_size Size of comrpessed data.
//!\param[out] decompressed_data Decompressed data.
//!\param[in] config For configuring the decompression process.
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, Config config);

//! Verify the compressed data
bool verify(const void *compressed_data, size_t compressed_size);

//! Query the data type of original data
enum data_type infer_data_type(const void *compressed_data,
                               size_t compressed_size);

//! Query the shape of original data
std::vector<SIZE> infer_shape(const void *compressed_data,
                              size_t compressed_size);

//! Query the data structure of original data
enum data_structure_type infer_data_structure(const void *compressed_data,
                                              size_t compressed_size);

//! Query the file used to store the coordinates data
std::string infer_nonuniform_coords_file(const void *compressed_data,
                                         size_t compressed_size);

//! Query the coordinates
template <typename T>
std::vector<T *> infer_coords(const void *compressed_data,
                              size_t compressed_size);

} // namespace mgard_x

#endif