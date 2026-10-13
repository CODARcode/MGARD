/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: October 5, 2026
 */

#ifndef MGARD_X_DATA_COMPATIBILITY_H
#define MGARD_X_DATA_COMPATIBILITY_H

#include <vector>

#include "../Config/Config.h"

namespace mgard_x {

// Whether compressed_data was written by MGARD 1.6.x or earlier, read from the
// software version in its header. Such streams are decompressed by the
// separate MGARD 1.6.0-compatible pipeline (include/mgard-x-160,
// src/mgard-x-160) when the library is built with
// MGARD_ENABLE_DATA_COMPATIBILITY.
template <typename DeviceType>
bool compressed_by_v160(const void *compressed_data);

namespace v160 {

// Entry points of the MGARD 1.6.0-compatible decompression pipeline. Same
// contracts as the corresponding mgard_x::decompress overloads; only defined
// when the library is built with MGARD_ENABLE_DATA_COMPATIBILITY.
template <typename DeviceType>
enum compress_status_type
decompress(const void *compressed_data, size_t compressed_size,
           void *&decompressed_data, Config config, bool output_pre_allocated);

template <typename DeviceType>
enum compress_status_type
decompress(const void *compressed_data, size_t compressed_size,
           void *&decompressed_data, data_type &dtype, std::vector<SIZE> &shape,
           Config config, bool output_pre_allocated);

template <typename DeviceType> enum compress_status_type release_cache();

} // namespace v160

} // namespace mgard_x

#endif
