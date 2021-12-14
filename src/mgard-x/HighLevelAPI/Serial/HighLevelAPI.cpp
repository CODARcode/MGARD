/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#define MGARDX_COMPILE_SERIAL
#include "mgard-x/HighLevelAPI.hpp"

#include <iostream>

#include <chrono>
namespace mgard_x {

#define KERNELS(D, T)                                        \
template void compress<D, T, Serial>(std::vector<SIZE> shape, T tol, T s, enum error_bound_type mode,\
              const void *original_data, void *&compressed_data,\
              size_t &compressed_size, Config config, bool output_pre_allocated);\
template void compress<D, T, Serial>(std::vector<SIZE> shape, T tol, T s, enum error_bound_type mode,\
              const void *original_data, void *&compressed_data,\
              size_t &compressed_size, Config config, std::vector<T *> coords, bool output_pre_allocated);\
template void decompress<D, T, Serial>(std::vector<SIZE> shape, const void *compressed_data,\
                size_t compressed_size, void *&decompressed_data,\
                std::vector<T *> coords, Config config, bool output_pre_allocated);\
template void decompress<D, T, Serial>(std::vector<SIZE> shape, const void *compressed_data,\
                size_t compressed_size, void *&decompressed_data,\
                Config config, bool output_pre_allocated);

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)
#undef KERNELS

template void BeginAutoTuning<Serial>();
template void EndAutoTuning<Serial>();
} // namespace mgard_x
#undef MGARDX_COMPILE_SERIAL