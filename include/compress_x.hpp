/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "mgard-x/CompressionWorkflow.h"
#include "mgard-x/RuntimeX/RuntimeXPublic.h"
#include "mgard-x/Types.h"
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

//! Enable autotuning
void BeginAutoTuning(enum device_type dev_type);

//! Disable autotuning
void EndAutoTuning(enum device_type dev_type);

//!\file
//!\brief Low level compression and decompression API.

//! Compress a function on an N-D tensor product grid
//!
//!\param[in] hierarchy Hierarchy type for storing precomputed variable to
//! help speed up compression.
//!\param[in] in_array Dataset to be compressed.
//!\param[in] type Error bound type: REL or ABS.
//!\param[in] tol Relative error tolerance.
//!\param[in] s Smoothness parameter to use in compressing the function.
//!
//!\return Compressed dataset.
template <uint32_t D, typename T, typename DeviceType>
Array<1, unsigned char, DeviceType>
compress(Hierarchy<D, T, DeviceType> &hierarchy,
         Array<D, T, DeviceType> &in_array, enum error_bound_type type, T tol,
         T s);

//! Decompress a function on an N-D tensor product grid
//!
//!\param[in] hierarchy Hierarchy type for storing precomputed variable to
//! help speed up decompression.
//!\param[in] compressed_array Compressed dataset.
//!\return Decompressed dataset.
template <uint32_t D, typename T, typename DeviceType>
Array<D, T, DeviceType>
decompress(Hierarchy<D, T, DeviceType> &hierarchy,
           Array<1, unsigned char, DeviceType> &compressed_array);

//! Enable autotuning
template <typename DeviceType> void BeginAutoTuning();

//! Disable autotuning
template <typename DeviceType> void EndAutoTuning();

} // namespace mgard_x

#endif