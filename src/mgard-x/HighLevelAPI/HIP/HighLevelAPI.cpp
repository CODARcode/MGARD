/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "mgard-x/HighLevelAPI.hpp"

#include <iostream>

#include <chrono>
namespace mgard_x {

template void compress<HIP>(DIM D, data_type dtype, std::vector<SIZE> shape,
                            double tol, double s, enum error_bound_type mode,
                            const void *original_data, void *&compressed_data,
                            size_t &compressed_size, Config config,
                            bool output_pre_allocated);

template void compress<HIP>(DIM D, data_type dtype, std::vector<SIZE> shape,
                            double tol, double s, enum error_bound_type mode,
                            const void *original_data, void *&compressed_data,
                            size_t &compressed_size, bool output_pre_allocated);

template void compress<HIP>(DIM D, data_type dtype, std::vector<SIZE> shape,
                            double tol, double s, enum error_bound_type mode,
                            const void *original_data, void *&compressed_data,
                            size_t &compressed_size,
                            std::vector<const Byte *> coords, Config config,
                            bool output_pre_allocated);

template void compress<HIP>(DIM D, data_type dtype, std::vector<SIZE> shape,
                            double tol, double s, enum error_bound_type mode,
                            const void *original_data, void *&compressed_data,
                            size_t &compressed_size,
                            std::vector<const Byte *> coords,
                            bool output_pre_allocated);

template void decompress<HIP>(const void *compressed_data,
                              size_t compressed_size, void *&decompressed_data,
                              Config config, bool output_pre_allocated);

template void decompress<HIP>(const void *compressed_data,
                              size_t compressed_size, void *&decompressed_data,
                              bool output_pre_allocated);

template void decompress<HIP>(const void *compressed_data,
                              size_t compressed_size, void *&decompressed_data,
                              data_type &dtype,
                              std::vector<mgard_x::SIZE> &shape, Config config,
                              bool output_pre_allocated);

template void decompress<HIP>(const void *compressed_data,
                              size_t compressed_size, void *&decompressed_data,
                              data_type &dtype,
                              std::vector<mgard_x::SIZE> &shape,
                              bool output_pre_allocated);

template void BeginAutoTuning<HIP>();
template void EndAutoTuning<HIP>();

} // namespace mgard_x