/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// #include "../Hierarchy/Hierarchy.h"
// #include "../RuntimeX/RuntimeXPublic.h"
// #include "Metadata.hpp"

#ifndef MGARD_X_160_COMPRESSION_HIGH_LEVEL_API_H
#define MGARD_X_160_COMPRESSION_HIGH_LEVEL_API_H

namespace mgard_x_160 {

template <typename DeviceType>
enum compress_status_type
compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol, double s,
         enum error_bound_type mode, const void *original_data,
         void *&compressed_data, size_t &compressed_size, Config config,
         bool output_pre_allocated);

template <typename DeviceType>
enum compress_status_type
compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol, double s,
         enum error_bound_type mode, const void *original_data,
         void *&compressed_data, size_t &compressed_size,
         bool output_pre_allocated);

template <typename DeviceType>
enum compress_status_type
compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol, double s,
         enum error_bound_type mode, const void *original_data,
         void *&compressed_data, size_t &compressed_size,
         std::vector<const Byte *> coords, Config config,
         bool output_pre_allocated);

template <typename DeviceType>
enum compress_status_type
compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol, double s,
         enum error_bound_type mode, const void *original_data,
         void *&compressed_data, size_t &compressed_size,
         std::vector<const Byte *> coords, bool output_pre_allocated);

template <typename DeviceType>
enum compress_status_type
decompress(const void *compressed_data, size_t compressed_size,
           void *&decompressed_data, Config config, bool output_pre_allocated);

template <typename DeviceType>
enum compress_status_type
decompress(const void *compressed_data, size_t compressed_size,
           void *&decompressed_data, bool output_pre_allocated);

template <typename DeviceType>
enum compress_status_type decompress(const void *compressed_data,
                                     size_t compressed_size,
                                     void *&decompressed_data, data_type &dtype,
                                     std::vector<mgard_x_160::SIZE> &shape,
                                     Config config, bool output_pre_allocated);

template <typename DeviceType>
enum compress_status_type
decompress(const void *compressed_data, size_t compressed_size,
           void *&decompressed_data, data_type &dtype,
           std::vector<mgard_x_160::SIZE> &shape, bool output_pre_allocated);

template <typename DeviceType> enum compress_status_type release_cache();

template <typename DeviceType> void pin_memory(void *ptr, SIZE num_bytes);

template <typename DeviceType> bool check_memory_pinned(void *ptr);

template <typename DeviceType> void unpin_memory(void *ptr);

} // namespace mgard_x_160

#endif